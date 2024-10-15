import os
import argparse
import abnumber
import pyrosetta; pyrosetta.init()
from pyrosetta import rosetta
from Bio.PDB import PDBParser, PDBIO

_NB_CHAIN = "A"
_PACK_RADIUS = 4.5
_N_RELAX = 2
_NB_NUMBERING_SCHEME = "imgt"

# inputs
parser = argparse.ArgumentParser(description="Sample nanobody loops")
parser.add_argument("pdb_file", help="PDB file containing the nanobody.")
parser.add_argument("-m", "--mutations", nargs="+", default=[],
                    help="Point mutations")
parser.add_argument("-o", "--output_path", default=".", help="Output path")
parser.add_argument("-n", "--n_samples", type=int, default=8, 
                    help="Number of conformations")

args = parser.parse_args()
input_pdb_file = os.path.abspath(args.pdb_file)
mutations = args.mutations
output_path = os.path.abspath(args.output_path)
n_samples = args.n_samples

os.system(f"mkdir -p {output_path}")

# score function
sfxn = pyrosetta.create_score_function("ref2015")

# read nanobody structure into a pose
pose = pyrosetta.pose_from_pdb(input_pdb_file)

# apply given mutations
for m in mutations:
    pdb_res, to_aa = int(m[1:-1]), m[-1]
    pose_res = pose.pdb_info().pdb2pose(_NB_CHAIN, pdb_res)
    pyrosetta.toolbox.mutants.mutate_residue(
        pose, pose_res, to_aa, 
        pack_radius=_PACK_RADIUS,
        pack_scorefxn=sfxn)

# get the CDR3 region to refine
chain = abnumber.Chain(pose.sequence(), scheme=_NB_NUMBERING_SCHEME)
subseq = chain.cdr3_seq
start = pose.sequence().find(subseq) - 1
# cutpoint is chosen in a way to preserve helices, if any in the CDR3
cutpoint = start + 1 + len(subseq)
stop = cutpoint + 2
loop = rosetta.protocols.loops.Loop(start, stop, cutpoint)

# sample loop conformations and relax them
io = PDBIO()
fastrelax = rosetta.protocols.relax.FastRelax(sfxn, _N_RELAX)
kic = rosetta.protocols.kinematic_closure.KicMover()
kic.set_loop(loop)

for i in range(n_samples):
    print("\nTrial", i)
    if i > 0: kic.apply(pose)
    fastrelax.apply(pose)
    
    this_pdb_file = os.path.join(output_path, f"conf_{i}.pdb")
    pose.dump_pdb(this_pdb_file)
    
    model = PDBParser(QUIET=True).get_structure("x", this_pdb_file)[0]
    io.set_structure(model)
    io.save(this_pdb_file)
