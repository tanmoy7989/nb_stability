import numpy as np
import itertools

import mdtraj
import abnumber

from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1 as seq3to1

from numba import njit, prange, types

N_BINS = 50
EPS = 1e-10
BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def get_sequence_region_boundaries(pdb_file, chain="A"):
    model = PDBParser(QUIET=True).get_structure("x", pdb_file)[0][chain]
    residues = sorted(list(model.get_residues()), key=lambda r: int(r.id[1]))
    seq = "".join([seq3to1(r.resname) for r in residues])
    
    _boundary = lambda s, s0: (s.find(s0), s.find(s0) + len(s0))
    
    chain = abnumber.Chain(seq, scheme="imgt")
    boundaries = {
        "fr1": _boundary(seq, chain.fr1_seq),
        "fr2": _boundary(seq, chain.fr2_seq),
        "fr3": _boundary(seq, chain.fr3_seq),
        "fr4": _boundary(seq, chain.fr4_seq),
        "cdr1": _boundary(seq, chain.cdr1_seq),
        "cdr2": _boundary(seq, chain.cdr2_seq),
        "cdr3": _boundary(seq, chain.cdr3_seq)
    }    
    return seq, boundaries


def read_trajectory(traj_files, ref_pdb, frames_per_run):
    trjs = []
    for f in traj_files:
        trj = mdtraj.load(f, top=ref_pdb)
        trjs.append(trj[-frames_per_run:])
    return mdtraj.join(trjs)


def embed_residue_bfactors(in_pdb_file, out_pdb_file, bfactors, chain="A"):
    model = PDBParser(QUIET=True).get_structure("x", in_pdb_file)[0][chain]
    for i, r in enumerate(list(model.get_residues())):
        bf = bfactors[i]
        [a.set_bfactor(bf) for a in r.get_atoms()]
        
    io = PDBIO()
    io.set_structure(model)
    io.save(out_pdb_file)


# ---------------------------
# STRUCTURAL ORDER PARAMETERS
# ---------------------------
def get_backbone_RMSD(trj, ref_trj, resids=[]):
    if not resids:
        resids = [r.resSeq for r in ref_trj.topology.residues]
    atom_indices = []
    for r in ref_trj.topology.residues:
        if r.resSeq not in resids:
            continue
        this_atom_indices = [a.index for a in r.atoms
                             if a.name in BACKBONE_ATOMS]
        atom_indices.extend(this_atom_indices)
    return mdtraj.rmsd(trj, ref_trj, atom_indices=atom_indices)


def get_per_residue_backbone_RMSD(trj, ref_trj):
    # superpose with reference based on heavy atoms only
    heavy_indices = ref_trj.topology.select("backbone")
    ref_trj.center_coordinates()
    trj.superpose(ref_trj, atom_indices=heavy_indices)
    
    # get avg dRMS for each residue
    rmsf = []
    for r in ref_trj.topology.residues:
        atom_indices = [a.index for a in r.atoms if a.name in BACKBONE_ATOMS]
        xyz = trj.atom_slice(atom_indices).xyz
        xyz_ref = ref_trj.atom_slice(atom_indices).xyz
        
        dxyz = (xyz-xyz_ref)*(xyz-xyz_ref)
        msd = np.mean(dxyz.reshape(dxyz.shape[0], -1), axis=1)
        rmsf.append(np.sqrt(msd))
    
    return np.array(rmsf).transpose()


def get_Q(trj, ref_trj, resids=[]):
    # adapted from: https://mdtraj.org/1.9.3/examples/native-contact.html
    beta_const = 50 # 1/nm
    lambda_const = 1.8 # all atom model
    native_cutoff = 0.45 # nm
    
    # get indices of all heavy atom
    if not resids:
        resids = [r.resSeq for r in ref_trj.topology.residues]
    heavy_indices_ = ref_trj.topology.select_atom_indices("heavy")
    heavy_indices = [i for i in heavy_indices_ 
                     if ref_trj.topology.atom(i).residue.resSeq in resids]
    
    # get all heavy atom pairs
    heavy_pairs = []
    for (i, j) in itertools.combinations(heavy_indices, 2):
        gap = abs(ref_trj.topology.atom(i).residue.index - 
                  ref_trj.topology.atom(j).residue.index)
        if gap > 3: heavy_pairs.append((i,j))
    heavy_pairs = np.array(heavy_pairs)
    
    # get contact atom pairs
    heavy_pairs_distances = mdtraj.compute_distances(ref_trj[0], heavy_pairs)[0]
    contact_pairs = heavy_pairs[heavy_pairs_distances <= native_cutoff]
    
    # now compute distances
    r = mdtraj.compute_distances(trj, contact_pairs)
    r0 = mdtraj.compute_distances(ref_trj[0], contact_pairs)
    q = 1.0 / (1 + np.exp( beta_const * (r - lambda_const*r0) ))
    
    return np.mean(q, axis=1)


def get_phipsi_deviations(trj, ref_trj, resids=[]):
    if not resids:
        resids = [r.resSeq for r in ref_trj.topology.residues]
    trj.superpose(ref_trj, atom_indices=ref_trj.topology.select("backbone"))
    
    nframes = len(trj)
    nres = len(list(ref_trj.topology.residues))
    indices = [r.index for r in ref_trj.topology.residues
               if r.resSeq in resids]
    
    # compute phi and zero pad
    _, phi = mdtraj.compute_phi(trj)
    _, phi_ref = mdtraj.compute_phi(ref_trj)
    dphi = np.arctan2(np.sin(phi-phi_ref), np.cos(phi-phi_ref))
    if 0 in indices:
        dphi = np.hstack([np.zeros(nframes).reshape(-1,1), dphi]) 
    dphi = dphi[:, indices]
    
    # compute phi and zero pad
    _, psi = mdtraj.compute_psi(trj)
    _, psi_ref = mdtraj.compute_psi(ref_trj)
    dpsi = np.arctan2(np.sin(psi-psi_ref), np.cos(psi-psi_ref))
    if nres-1 in indices:
        dpsi = np.hstack([dpsi, np.zeros(nframes).reshape(-1,1)]) 
    dpsi = dpsi[:, indices]

    return np.sqrt(dphi*dphi + dpsi*dpsi)
    
    
# ----------------------------------------
# NUMBA-ized MUTUAL INFORMATION CALCULATON
# ----------------------------------------
# parallelized self entropy calculator
@njit(
    types.float32(
        types.Array(types.float32, 1, "C"), 
        types.int64, 
        types.float32, 
        types.float32
    ), 
    parallel=True, fastmath=True
)
def _entropy_1D(
    x: types.Array(types.float32, 1, "C"), 
    nbins: types.int64, 
    xmin: types.float32, 
    xmax: types.float32
) -> types.float32:
    
    hist = np.zeros(nbins, dtype=np.int64)
    bin_width = (xmax - xmin) / nbins
    inv_bin_width = 1.0 / bin_width

    for i in prange(x.shape[0]):
        bin_idx = int((x[i] - xmin) * inv_bin_width)
        if bin_idx >= 0 and bin_idx < nbins:
            hist[bin_idx] += 1

    hist = hist.astype(np.float32) / np.sum(hist) + np.float32(EPS)
    entropy = -np.sum(hist * np.log(hist))
    return entropy


# parallelized cross entropy calculator
@njit(
    types.float32(
        types.Array(types.float32, 2, "C"), 
        types.int64, 
        types.float32, 
        types.float32
    ), 
    parallel=True, fastmath=True
)
def _entropy_2D(
    x: types.Array(types.float32, 2, "C"), 
    nbins: types.int64, 
    xmin: types.float32, 
    xmax: types.float32
) -> types.float32:
    
    hist = np.zeros((nbins, nbins), dtype=np.int32)
    bin_width = (xmax - xmin) / nbins
    inv_bin_width = 1.0 / bin_width
    
    for i in prange(x.shape[0]):
        bin_idx_x = int((x[i, 0] - xmin) * inv_bin_width)
        bin_idx_y = int((x[i, 1] - xmin) * inv_bin_width)
        if (bin_idx_x >= 0 and bin_idx_x < nbins and \
            bin_idx_y  >=0 and bin_idx_y < nbins):
            hist[bin_idx_x, bin_idx_y] += 1
                
    hist = hist.astype(np.float32) / np.sum(hist) + np.float32(EPS)
    entropy = -np.sum(hist * np.log(hist))
    return entropy


# parallelized trajectory spanning entropy calculator
@njit(
    types.Tuple((
        types.Array(types.float32, 1, "C"),
        types.Array(types.float32, 1, "C")
        ))(
            types.Array(types.float32, 2, "C"),
            types.int64,
            types.float32,
            types.float32
        ),
    parallel=True, fastmath=True
)
def _get_entropies_across_traj(
    x: types.Array(types.float32, 2, "C"), 
    nbins: types.int64,
    xmin: types.float32,
    xmax: types.float32
)-> types.Tuple((
    types.Array(types.float32, 1, "C"),
    types.Array(types.float32, 1, "C")
)):
    
    nres = x.shape[1]
    
    # get self_entropies
    self_entropies = np.zeros(nres, np.float32)
    for ii in prange(nres): 
        self_entropies[ii] = _entropy_1D(x[:, ii].flatten(), nbins, xmin, xmax)
    
    # get cross entropies
    cross_entropies = np.zeros(nres*nres, np.float32)
    for k in prange(nres*nres):
        i = int(k/nres)
        j = k - i*nres
        if i < j:
            this_x = np.array(list(zip(x[:, i], x[:, j]))).astype(np.float32)
            cross_entropies[k] = _entropy_2D(this_x, nbins, xmin, xmax)
    
    return self_entropies, cross_entropies


def get_mutual_information(x, nbins, xmin, xmax):
    nres = x.shape[1]
    
    # cast appropriately
    x = np.ascontiguousarray(x, dtype=np.float32)
    xmin, xmax = np.float32(xmin), np.float32(xmax)
    
    Hx_, Hxy_ = _get_entropies_across_traj(x, nbins, xmin, xmax)
    
    Hx = np.full((nres, nres), Hx_)
    Hy = Hx.transpose()
    
    Hxy = Hxy_.reshape(nres, nres)
    Hxy = Hxy + Hxy.transpose()
    
    I =  Hx + Hy - Hxy
    C = np.sqrt(1.0 - np.exp(-2*I/3))
        
    return I, C
