import os
import argparse
import numpy as np

from pdbfixer import PDBFixer
from openmm import unit, Vec3, \
        Platform, \
        MonteCarloBarostat, CustomExternalForce, \
        LangevinMiddleIntegrator
        
from openmm.app import HBonds, PME, NoCutoff, \
        PDBFile, ForceField, Modeller, Simulation, \
        DCDReporter, StateDataReporter

import mdtraj

# configuration
CONFIG = {
    "protein_ff": "amber14/protein.ff14SB.xml",
    "nonbonded_cutoff": 1.0 * unit.nanometer,
    "restraint_force_const": 1.0 * unit.kilocalorie_per_mole / (unit.angstrom)**2,
    "hydrogen_mass": 4.0 * unit.amu,

    "water_ff": "amber14/tip3p.xml",
    "water_model": "tip3p",
    "waterbox_pad": 0.2 * unit.nanometer,
    "pH": 7.4,
    "temp": 280.0 * unit.kelvin,
    "n_anneal_stages": 10,
    
    "timestep": 2.0 * unit.femtosecond,
    "collision_rate": 1.0 / unit.picosecond,
    "press": 1.013 * unit.bar,
    
    "platform": Platform.getPlatformByName("CUDA"),
    "platform_prop": {
        "DeviceIndex": "0", 
        "Precision": "mixed"
    }
}


N_STEPS = {
    "relax": 10000,
    "npt_with_restraint": 2000000,
    "npt_without_restraint": 1000000,
    "npt_freq": 1000,
    "npt_boxl": 4000,
    "nvt_anneal": 500000,
    "nvt": 100000000,
    "nvt_freq": 10000
}


def _setup(topology, ff, extra_forces=[], ff_kwargs=dict(), 
           positions=None, temp=None):
    
    all_ff_kwargs = {
        "hydrogenMass": CONFIG["hydrogen_mass"],
        "constraints": HBonds,
        "removeCMMotion": True
    }
    
    if ff_kwargs:
        all_ff_kwargs.update(ff_kwargs)
        
    system = ff.createSystem(topology, **all_ff_kwargs)
               
    if extra_forces:
        for f in extra_forces: system.addForce(f)
    
    if temp is None:
        temp = CONFIG["temp"]
    integrator = LangevinMiddleIntegrator(temp,
                                          CONFIG["collision_rate"],
                                          CONFIG["timestep"])
    
    simulation = Simulation(
            topology, system, integrator, 
            platform=CONFIG["platform"],
            platformProperties=CONFIG["platform_prop"]
        )
        
    if positions is not None:
        simulation.context.setPositions(positions)
    
    return simulation


def _write(simulation, out_pdb_file):
    pos = simulation.context.getState(
        getPositions=True, enforcePeriodicBox=False).getPositions()
    
    with open(out_pdb_file, "w") as of:
        PDBFile.writeFile(simulation.topology, pos, file=of, keepIds=True)


def _process_solvated_trj(trj, remove_solvent=False):
    # center
    mols_solute = trj.topology.find_molecules()[0:1]
    trj = trj.image_molecules(anchor_molecules=mols_solute)
    # remove solvent
    if remove_solvent:
        trj = trj.atom_slice(trj.topology.select("chainid == 0"))
    return trj
        

def clean_pdb(in_pdb_file, out_prefix):
    # filename
    out_pdb_file = out_prefix + ".relaxed.pdb"
    
    # fixed topology
    print("Fixing input pdb topology")
    p = PDBFixer(in_pdb_file)
    p.findMissingResidues()
    p.findMissingAtoms()
    p.addMissingAtoms()
    p.addMissingHydrogens(pH=CONFIG["pH"])
    
    # forcefield
    ff = ForceField(CONFIG["protein_ff"])
    ff_kwargs = {"nonbondedMethod": NoCutoff}
    
    # energy minimization
    sim = _setup(p.topology, ff, ff_kwargs=ff_kwargs, positions=p.positions)
    sim.minimizeEnergy(maxIterations=N_STEPS["relax"])
    _write(sim, out_pdb_file)
    return out_pdb_file


def solvate(in_pdb_file, out_prefix, rng_seed=85431):
    # filenames
    out_pdb_file = out_prefix + ".npt-solvated.pdb"
    out_traj_file = out_prefix + ".npt-solvated.dcd"
    out_boxlen_file = out_prefix + ".boxl.txt"
    
    # forcefield
    ff = ForceField(CONFIG["protein_ff"], CONFIG["water_ff"])
    ff_kwargs = {"nonbondedMethod": PME, 
                 "nonbondedCutoff": CONFIG["nonbonded_cutoff"]}
    
    # solvated topology
    p = PDBFixer(in_pdb_file)
    m = Modeller(p.topology, p.positions)
    m.addSolvent(ff, model=CONFIG["water_model"],
                 padding=CONFIG["waterbox_pad"])

    # barostat for NPT sims (2 instances for 2 separate sims)
    f_barostat = [
        MonteCarloBarostat(CONFIG["press"], CONFIG["temp"]),
        MonteCarloBarostat(CONFIG["press"], CONFIG["temp"])
    ]
    
    # position restraint
    f_posres = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    f_posres.addGlobalParameter("k", CONFIG["restraint_force_const"])
    f_posres.addPerParticleParameter("x0")
    f_posres.addPerParticleParameter("y0")
    f_posres.addPerParticleParameter("z0")
    
    # equlibrate with restraint
    print("Equilbrating in NPT with position restraints...")
    sim1 = _setup(m.topology, ff, ff_kwargs=ff_kwargs,
                  extra_forces=[f_barostat[0], f_posres], 
                  positions=m.positions)
    
    sim1.minimizeEnergy(maxIterations=N_STEPS["relax"])
    sim1.context.setVelocitiesToTemperature(CONFIG["temp"], rng_seed)
    sim1.step(N_STEPS["npt_with_restraint"])
    pos1 = sim1.context.getState(
        getPositions=True, enforcePeriodicBox=False).getPositions()
    
    # equlibrate without restraint
    print("Equilbrating in NPT after removing position restraints...")
    sim2 = _setup(m.topology, ff, ff_kwargs=ff_kwargs,
                  extra_forces=[f_barostat[1]], positions=pos1)
    sim2.context.setPositions(pos1)
    sim2.minimizeEnergy(maxIterations=N_STEPS["relax"])
    _write(sim2, out_pdb_file)
    
    trj_reporter = DCDReporter(out_traj_file,
                               reportInterval=N_STEPS["npt_freq"],enforcePeriodicBox=False)
    sim2.reporters.extend([trj_reporter])
    sim2.step(N_STEPS["npt_without_restraint"])
    
    # get boxlength
    trj = mdtraj.load(out_traj_file, top=out_pdb_file)
    boxls = trj.unitcell_lengths[-N_STEPS["npt_boxl"] : ]
    boxl = np.mean(boxls, axis=0)
    np.savetxt(out_boxlen_file, boxl)
    
    # get last equilibrated frame
    trj = _process_solvated_trj(trj)
    trj[-1].save_pdb(out_pdb_file)
    return out_pdb_file, out_boxlen_file


def anneal(simulation, target_temp):
    tmin = CONFIG["temp"].value_in_unit(unit.kelvin)
    tmax = target_temp.value_in_unit(unit.kelvin)
    
    temp_schedule = np.logspace(
        np.log10(tmin), np.log10(tmax), CONFIG["n_anneal_stages"]
    ) * unit.kelvin
    
    for t in temp_schedule[1:]:
        print("> Raising temperature to: ", t)
        simulation.integrator.setTemperature(t)
        simulation.step(N_STEPS["nvt_anneal"])
        
    return simulation


def run_md(in_pdb_file, boxlen_file, out_prefix, target_temp, rng_seed=85431):
    print("NVT: Setting up system...")
    
    # filenames
    out_traj_file = out_prefix + ".nvt-solvated.dcd"
    out_dry_traj_file = out_prefix + ".nvt.dcd"
    out_energy_file = out_prefix + ".nvt-energy.txt"
    
    # forcefield
    ff = ForceField(CONFIG["protein_ff"], CONFIG["water_ff"])
    ff_kwargs = {"nonbondedMethod": PME, 
                 "nonbondedCutoff": CONFIG["nonbonded_cutoff"]}
    
    # prepare simulation
    p = PDBFile(in_pdb_file)
    sim = _setup(p.topology, ff, ff_kwargs=ff_kwargs, positions=p.positions,
                 temp=CONFIG["temp"])
    
    # set box vectors
    L = np.loadtxt(boxlen_file)
    sim.context.setPeriodicBoxVectors(
            a=Vec3(L[0], 0, 0), b=Vec3(0, L[1], 0), c=Vec3(0, 0, L[2])
    )

    # energy minimize
    print("NVT: Energy minimizing...")
    sim.minimizeEnergy(maxIterations=N_STEPS["relax"])

    # heat up the system slowly
    if not np.isclose(
        target_temp.value_in_unit(unit.kelvin),
        CONFIG["temp"].value_in_unit(unit.kelvin)
    ):
        print("NVT: Annealing to target temperature...")
        anneal(sim, target_temp)
        
    # add trajectory and energy reporter
    trj_reporter = DCDReporter(out_traj_file,
                               reportInterval=N_STEPS["nvt_freq"],enforcePeriodicBox=False)
    
    st_reporter = StateDataReporter(out_energy_file,
                                reportInterval=N_STEPS["nvt_freq"],
                                step=True, temperature=True,
                                potentialEnergy=True)

    sim.reporters.extend([trj_reporter, st_reporter])

    # run production NVT 
    print("NVT: Running production...")
    sim.context.setVelocitiesToTemperature(target_temp, rng_seed)
    sim.step(N_STEPS["nvt"])
    
    # remove solvent and center traj
    trj = mdtraj.load(out_traj_file, top=in_pdb_file)
    trj = _process_solvated_trj(trj, remove_solvent=True)
    trj.save(out_dry_traj_file)
    

def main():
    parser = argparse.ArgumentParser(description="Nanobody stability MD sims")
    parser.add_argument("pdb_file", 
            help="PDB file containing the nanobody.")
    parser.add_argument("-t", "--target_temp", type=float, default=300,
            help="Target temperature.")
    parser.add_argument("-o", "--out_prefix", default="nb",
            help="Output prefix for all outputs. Can include the output path.")
    parser.add_argument("-r", "--random_seed", type=int, default=54321, 
            help="Random number seed for setting velocities")

    # parse inputs
    args = parser.parse_args()
    in_pdb_file = os.path.abspath(args.pdb_file)
    target_temp = args.target_temp * unit.kelvin
    out_prefix = os.path.abspath(args.out_prefix)
    rng_seed = args.random_seed
    
    # make output path
    output_path = os.path.dirname(out_prefix)
    os.system(f"mkdir -p {output_path}")
    
    # clean the pdb file
    cleaned_pdb_file = clean_pdb(in_pdb_file, out_prefix)
    
    # solvate
    solvated_pdb_file, boxl_file = solvate(cleaned_pdb_file, out_prefix,
                                           rng_seed=rng_seed)
    
    # run expanded-ensemble simulations in NVT + temperature space.
    run_md(solvated_pdb_file, boxl_file, out_prefix, target_temp, 
           rng_seed=rng_seed)
    
    
if __name__ == "__main__":
    main()
