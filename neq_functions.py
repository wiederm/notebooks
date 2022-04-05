# general imports
import pickle
import random

import mdtraj as md
import numpy as np
# Imports from the openff toolkit
import openff.toolkit
import torch
from mdtraj import Trajectory
from openff.toolkit.typing.engines.smirnoff import ForceField
from torchani.models import ANI2x
from tqdm import tqdm

forcefield = ForceField('openff_unconstrained-2.0.0.offxml')
print(openff.toolkit._version.get_versions())

# Imports from openMM
from NNPOps import OptimizedTorchANI
from openmm import LangevinIntegrator, Platform, unit
from openmm.app import Simulation

##################################
hartree_to_kJ_mol = 2625.5
mass_dict_in_daltons = {"H": 1.0, "C": 12.0, "N": 14.0, "O": 16.0}
distance_unit = unit.angstrom
time_unit = unit.femtoseconds
speed_unit = distance_unit / time_unit
energy_unit = unit.kilojoule_per_mole
stepsize = 1 * time_unit
collision_rate = 1 / unit.picosecond
temperature = 300 * unit.kelvin
from openmmtools.constants import kB
kBT = kB * temperature
##################################


class NNP(torch.nn.Module):
    
    def __init__(self, molecule, platform):
        super().__init__()
        self.platform = platform
        # Store the atomic numbers
        self.species = torch.tensor([[atom.atomic_number for atom in molecule.atoms]], device=self.platform)
        # Create an ANI-2x model
        self.model = ANI2x(periodic_table_index=True)
        # Accelerate the model
        self.model = OptimizedTorchANI(self.model, self.species).to(device=self.platform)
        # save atoms as string
        self.atoms = ''.join([a.element.symbol for a in molecule.atoms])

    def _calculate_energy(self, coordinates: torch.tensor):
        """
        Helpter function to return energies as tensor.
        Given a coordinate set the energy is calculated.
        Parameters
        ----------
        coordinates : torch.tensor 
            coordinates in angstrom without units attached
        Returns
        -------
        energy_in_kJ_mol : torch.tensor
        """
        #coordinates = coordinates.unsqueeze(0).float()
        energy_in_hartree = self.model((self.species, coordinates)).energies

        # convert energy from hartrees to kJ/mol
        return energy_in_hartree * hartree_to_kJ_mol
    
    def calculate_force(self, coordinates: unit.Quantity):
        """
        Given a coordinate set the forces with respect to the coordinates are calculated.
        Parameters
        ----------
        coordinates : numpy array in angstrom 
            initial configuration
        Returns
        -------
        F, E : float, in kJ/mol/A, kJ/mol
        """
        
        assert type(coordinates) is unit.Quantity
        coordinates_ = torch.tensor([coordinates.value_in_unit(unit.angstrom)],  dtype=torch.float32, requires_grad=True, device=self.platform)
        energy_in_kJ_mol = self._calculate_energy(coordinates_)

        # derivative of E (in kJ/mol) w.r.t. coordinates (in Angstrom)
        derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates_)[0]
        if self.platform == 'cpu':
            F = - np.array(derivative)
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return F*(unit.kilojoule_per_mole / unit.angstrom), energy_in_kJ_mol * unit.kilojoule_per_mole # kilojoule_per_mole / angstrom; kJ/mol
    
    
    def calculate_energy(self, coordinates: unit.Quantity):
        """
        Given a coordinate set the energy is calculated.
        Parameters
        ----------
        coordinates : unit'd 
            coordinates
        Returns
        -------
        energy : torch.tensor
        """

        assert type(coordinates) is unit.Quantity
        coordinates_ = torch.tensor([coordinates.value_in_unit(unit.angstrom)],  dtype=torch.float32, requires_grad=True, device=self.platform)
        return self._calculate_energy(coordinates_).detach().cpu().numpy() * unit.kilojoule_per_mole
    



class LangevinDynamics(object):
    def __init__(self, atoms: str, energy_and_force):

        self.energy_and_force = energy_and_force
        self.temperature = temperature
        self.atoms = atoms

    def run_dynamics(
        self,
        x0: np.ndarray,
        n_steps: int = 100,
        stepsize: unit.quantity.Quantity = 1.0 * unit.femtosecond,
        collision_rate: unit.quantity.Quantity = 10 /unit.picoseconds,
        progress_bar: bool = False,
        temperature=None
    ):
        """Unadjusted Langevin dynamics.
        Parameters
        ----------
        x0 : array of floats, in Angstrom
            initial configuration
        n_steps : integer
            number of Langevin steps
        stepsize : float > 0, in units of femtoseconds
        collision_rate : float > 0, in units of 1/ps
            controls the rate of interaction with the heat bath
        progress_bar : bool
            use tqdm to show progress bar
        Returns
        -------
        traj    : [n_steps + 1 x dim] array of floats, unit'd
            trajectory of samples generated by Langevin dynamics
        energy  : list of floats in kJ/mol
        stddev  : list of kJ/mol
        ensemble_bias : list of kJ/mol
        """
        assert type(x0) == unit.Quantity
        assert type(stepsize) == unit.Quantity
        assert type(collision_rate) == unit.Quantity

        if temperature == None:
            temperature = self.temperature

        assert type(temperature) == unit.Quantity

        # generate mass arrays
        masses = np.array([mass_dict_in_daltons[a] for a in self.atoms]) * unit.dalton
        sigma_v = np.array([unit.sqrt(kB * temperature / m ) / speed_unit for m in masses]) * speed_unit
        v0 = np.random.randn(len(sigma_v), 3) * sigma_v[:, None]
        # convert initial state numpy arrays with correct attached units
        x = np.array(x0.value_in_unit(distance_unit)) * distance_unit # x.shape =[N_atoms][3]
        v = np.array(v0.value_in_unit(speed_unit)) * speed_unit # v.shape = [N_atoms][3]
        # traj is accumulated as a list of arrays with attached units
        traj = [x]

        # dimensionless scalars
        a = np.exp(-collision_rate * stepsize)
        b = np.sqrt(1 - np.exp(-2 * collision_rate * stepsize))

        # compute force on initial configuration
        F, E, = self.energy_and_force.calculate_force(x)  # F.shape = [1][N_atoms][3]
        F = F[0] # to transform F.shape to [N_atoms][3]
        # energy is saved as a list
        energy = [E]

        trange = range(n_steps)
        if progress_bar:
            trange = tqdm(trange)

        # main loop
        for _ in trange:
            # v
            v += (stepsize * 0.5) * F / masses[:, None]
            # r
            x += (stepsize * 0.5) * v  # NOTE: x.shape = [n_atoms][3], but v.shape = [n_atoms][3]
            # o
            v = (a * v) + (b * sigma_v[:, None] * np.random.randn(*x.shape))
            # r
            x += (stepsize * 0.5) * v

            F, E = self.energy_and_force.calculate_force(x)
            F = F[0] # to transform F.shape to [N_atoms][3]
            energy.append(E)
            # v
            v += (stepsize * 0.5) * F / masses[:, None]

            norm_F = np.linalg.norm(F)
            # report gradient norm
            if progress_bar:
                trange.set_postfix({"|force|": norm_F})
            # check positions and forces are finite
            if (not np.isfinite(x).all()) or (not np.isfinite(norm_F)):
                print("Numerical instability encountered!")
                return traj, energy
            traj.append(x)

        return traj, energy
    
def save_traj(samples, molecule, name=''):
    molecule.to_file('test.pdb', file_format='pdb')
    top = md.load(f"test.pdb").topology
    traj = Trajectory(samples, topology=top)
    if name:        
        traj.save_dcd(f'{name}')
    else:
        traj.save_dcd('test.dcd')

    return traj
    
def create_mm_sim(molecule):
    """Create vacuum simulation system"""
    platform = Platform.getPlatformByName('CPU')
    properties={}
    properties["Threads"]="2"
    integrator = LangevinIntegrator(temperature, collision_rate, stepsize)  
    topology = molecule.to_topology()
    system = forcefield.create_openmm_system(topology)
    sim = Simulation(topology, system, integrator, platform=platform, platformProperties=properties)
    molecule.generate_conformers()
    sim.context.setPositions(molecule.conformers[0])
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature)
    return sim


def collect_samples_ani(molecule, n_samples=1_000, n_steps_per_sample=10_000, platform='cuda'):
    """generate samples using ANI2x"""
    
    print(f'Generate samples with QML: {n_samples=}, {n_steps_per_sample=}')
    energy_and_force = NNP(molecule, platform=platform)
    langevine = LangevinDynamics(energy_and_force.atoms,energy_and_force)
    # generate the position for ANI 
    positions = molecule.conformers[0]
    samples = []
    for _ in tqdm(range(n_samples)):
        samples, energy = langevine.run_dynamics(positions, n_steps_per_sample, stepsize=stepsize)
    return samples

def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


def collect_samples_mm(_, sim, n_samples=1_000, n_steps_per_sample=10_000):
    """generate samples using a classical FF"""
    
    print(f'Generate samples with MM: {n_samples=}, {n_steps_per_sample=}')   
    samples = []
    for _ in tqdm(range(n_samples)):
        sim.step(n_steps_per_sample)
        samples.append(get_positions(sim))
    return samples

def compute_mm_energy(sim, positions):
    """compute mm energy for given positions"""
    sim.context.setPositions(positions)
    return sim.context.getState(getEnergy=True).getPotentialEnergy()

def compute_mm_force(sim, positions):
    """compute mm forces given a position"""
    sim.context.setPositions(positions)
    return sim.context.getState(getForces=True).getForces(asNumpy=True)

def mixing(a, b, lambda_value):
    return ((1.-lambda_value) * a) + (lambda_value * b)   


def neq_from_ani_to_mm(molecule, 
                       n_samples:int, 
                       switching_length:int, 
                       n_steps_per_sample:int, 
                       nr_of_switches:int, 
                       save_samples:str = "", 
                       load_samples:str="", 
                       platform:str='cuda'):
    """NEQ switching from ANI to MM

    Args:
        molecule (_type_): _description_

    Returns:
        _type_: _description_
    """
    from functools import partial
    print('Performing NEQ switching from ANI to MM representation of molecule')
    
    # define systems
    energy_and_force = NNP(molecule, platform=platform)
    sim = create_mm_sim(molecule)
    # define energy and force calculation as a function of switching parameter lambda
    def _calculate_energy(x:unit.Quantity, lamb:float):
        assert type(x) == unit.Quantity
        assert lamb >= 0.0 and lamb <= 1.0
        ani_e = energy_and_force.calculate_energy(x)
        mm_e = compute_mm_energy(sim, x)
        return mixing(ani_e, mm_e, lamb)

    def _calculate_forces(x:unit.Quantity, lamb:float):
        assert type(x) == unit.Quantity
        assert lamb >= 0.0 and lamb <= 1.0
        ani_f = energy_and_force.calculate_force(x)[0]
        mm_f = compute_mm_force(sim, x)
        return mixing(ani_f[0], mm_f, lamb)
    
   
    # define function that collects samples 
    generate_samples = partial(collect_samples_ani, n_samples=n_samples, n_steps_per_sample = n_steps_per_sample)
       
    # call switching routine
    ws = _noneq_sampling_and_switching(nr_of_switches=nr_of_switches, 
                                  molecule=molecule, 
                                  generate_samples=generate_samples,
                                  calculate_force=_calculate_forces,
                                  calculate_energy = _calculate_energy,
                                  switching_length=switching_length,
                                  save_samples=save_samples,
                                  load_samples=load_samples
                                  )
   
    return ws


def neq_from_mm_to_ani(molecule, 
                       n_samples:int, 
                       n_steps_per_sample:int, 
                       nr_of_switches:int, 
                       switching_length:int, 
                       save_samples:str = "", 
                       load_samples:str="",
                       platform:str='cuda'):
    """NEQ switching from ANI to MM

    Args:
        molecule (_type_): _description_

    Returns:
        _type_: _description_
    """
    from functools import partial
    print('Performing NEQ switching from MM to ANI representation of molecule')
    
    # define systems
    energy_and_force = NNP(molecule, platform=platform)
    sim = create_mm_sim(molecule)
    # define energy and force calculation as a function of switching parameter lambda
    def _calculate_energy(x:unit.Quantity, lamb:float):
        assert type(x) == unit.Quantity
        assert lamb >= 0.0 and lamb <= 1.0
        ani_e = energy_and_force.calculate_energy(x)
        mm_e = compute_mm_energy(sim, x)
        return mixing(mm_e, ani_e, lamb)

    def _calculate_forces(x:unit.Quantity, lamb:float):
        assert type(x) == unit.Quantity
        assert lamb >= 0.0 and lamb <= 1.0
        ani_f = energy_and_force.calculate_force(x)[0]
        mm_f = compute_mm_force(sim, x)
        return mixing(mm_f, ani_f[0], lamb)
    
   
    # define function that collects samples 
    generate_samples = partial(collect_samples_mm, sim=sim, n_samples=n_samples, n_steps_per_sample = n_steps_per_sample)
       
    # call switching routine
    ws = _noneq_sampling_and_switching(nr_of_switches=nr_of_switches, 
                                  molecule=molecule, 
                                  generate_samples=generate_samples,
                                  calculate_force=_calculate_forces,
                                  calculate_energy = _calculate_energy,
                                  switching_length=switching_length,
                                  save_samples=save_samples,
                                  load_samples = load_samples
                                  )

    return ws

def _noneq_sampling_and_switching(
    nr_of_switches:int,
    molecule, 
    generate_samples, 
    calculate_force, 
    calculate_energy,
    switching_length:int,
    save_samples:str,
    load_samples:str,
    ):
    """
    Use nonequ switching to calculate work values from 'from_system' to 'to_system' starting with 'from_system' sampels.
    """
    lam_values = np.linspace(0,1,switching_length)      

    print(f'Perform NEQ using: {nr_of_switches=}, {switching_length=}')   
    
    # generate mass arrays
    atoms = ''.join([a.element.symbol for a in molecule.atoms])
    masses = np.array([mass_dict_in_daltons[a] for a in atoms]) * unit.daltons

    # dimensionless scalars
    a = np.exp(- collision_rate * stepsize)
    b = np.sqrt(1 - np.exp(-2 * collision_rate * stepsize))
    # generate sigma_v 
    sigma_v = np.array([unit.sqrt(kB * temperature / m) / speed_unit for m in masses]) * speed_unit

    # generate samples
    print('Start generating samples ...')
    if load_samples:
        print('Loading prgenerated samples  ...')
        samples = pickle.load(open(load_samples, 'rb'))
    else:
        samples = generate_samples(molecule)
        if save_samples:
            pickle.dump(samples, open(save_samples, 'wb+'))

    print('Samples generated ...')

    # w_list contains the work values for each switchin protocol
    w_list = []    
    print("Start with switching protocoll ...")
    for switch_nr in tqdm(range(nr_of_switches)):
        # traj accumulates the samples
        traj = []
        # select starting conformations
        x = np.array(random.choice(samples).value_in_unit(distance_unit)) * distance_unit
        # initial force
        F = calculate_force(x, lamb=0.)
        # seed velocities from boltzmann distribution
        v0 = np.random.randn(len(sigma_v), 3) * sigma_v[:, None]
        v = np.array(v0.value_in_unit(speed_unit)) * speed_unit

        w = 0.0  * unit.kilojoule_per_mole     
        for idx in range(1, switching_length):
            # v
            v += (stepsize * 0.5) * F / masses[:, None]
            # r
            x += (stepsize * 0.5) * v
            # o
            v = (a * v) + (b * sigma_v[:, None] * np.random.randn(*x.shape))
            # r
            x += (stepsize * 0.5) * v
            # calculate F
            F = calculate_force(x, lamb=lam_values[idx])
            # v
            v += (stepsize * 0.5) * F / masses[:, None]
            traj.append(x)
            # calculate work
            # evaluate u_t(x_t) - u_{t-1}(x_t)
            u_now = calculate_energy(x, lamb=lam_values[idx])
            u_before = calculate_energy(x, lamb=lam_values[idx-1])
            w += (u_now - u_before)
        
        w_list.append(w.value_in_unit(unit.kilojoule_per_mole))

        if save_samples:
            print(f'NEQ switching work: {w}')
            print('##################')
            save_traj(traj, molecule, name=f'{save_samples.split(".")[0]}_{switch_nr}.dcd')

    return np.array(w_list) * unit.kilojoule_per_mole


        