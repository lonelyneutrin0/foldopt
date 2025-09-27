"""Simulated annealing with the Metropolis criterion."""
from typing import Optional, List
import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike
from tqdm import tqdm

@dataclass
class RunData: 
    """Data class to store results of optimization runs."""
    energies: ArrayLike
    optimal_energy: ArrayLike
    accepts: ArrayLike
    rejects: ArrayLike
    temperatures: ArrayLike
    conformations: List

def metropolis_annealing(self, initial_temp: float, final_temp: float, cooling_rate: float, lam: float, chain_length: int, *, rng: Optional[np.random.Generator] = None) -> RunData:
    """Perform simulated annealing using the Metropolis criterion.

    Args:
        initial_temp (float): Starting temperature for annealing.
        final_temp (float): Ending temperature for annealing.
        cooling_rate (float): Rate at which the temperature decreases.
        lam (float): Perturbation factor. Higher values lead to larger distances
                     in the conformation space between current and proposed states.
        rng (np.random.Generator): Random number generator for reproducibility.
    Returns:
        RunData: Data from the optimization run.
    """
    
    num_iterations = (int)(np.log(final_temp / initial_temp) / np.log(cooling_rate))

    # Create arrays to store run data 
    energies = np.zeros(num_iterations)
    inv_temps = 1/initial_temp * np.power(1/cooling_rate, np.arange(num_iterations))
    accepts = np.zeros(num_iterations)
    rejects = np.zeros(num_iterations)
    total_proposals = num_iterations * chain_length
    rnds = rng.random(total_proposals) if rng else np.random.random(total_proposals)
    conformations = [self.conformation.copy()]
    current_energy = None 

    pbar = tqdm(range(num_iterations), desc=f"Energy: {current_energy}", unit="step", unit_scale=True)

    for step in pbar:
        current_energy = self.energy()
        pbar.set_description(f"Energy: {current_energy:.2f}")
        energies[step] = current_energy

        for i in range(chain_length):
            # Generate a new conformation by perturbing the current one
            self.propose(lam, ts=step/num_iterations, rng=rng)
            new_energy = self.energy()

            # Calculate energy difference
            delta_e = new_energy - current_energy
            rnd_idx = step * chain_length + i

            # Metropolis criterion - accept if energy is lower OR random chance
            if delta_e < 0 or rnds[rnd_idx] < np.exp(-inv_temps[step] * delta_e):
                # Accept the move - keep the new state
                current_energy = new_energy
                accepts[step] += 1
            else:
                # Reject the move - revert to previous state 
                self.revert()
                rejects[step] += 1
        
        conformations.append(self.conformation.copy())
            
    return RunData(
        energies=energies,
        optimal_energy=np.min(energies),
        accepts=accepts,
        rejects=rejects,
        temperatures=1/inv_temps,
        conformations=conformations
    )