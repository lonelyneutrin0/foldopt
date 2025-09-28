"""Hybrid genetic-simulated annealing for protein folding."""
import multiprocessing
import numpy as np 
from typing import Optional
from tqdm import tqdm
from ..opt.metropolis_annealing import RunData

def genetic_annealing(self, initial_temp: float, final_temp: float, cooling_rate: float, lam: float, population_size: int, crossover_rate: float, chain_length: int, rng: Optional[np.random.Generator] = None) -> RunData: 
    
    if rng is None:
        rng = np.random.default_rng()

    if population_size > multiprocessing.cpu_count():
        raise ValueError(f"Population size {population_size} exceeds available CPU cores {multiprocessing.cpu_count()}")
    
    population = []
    for _ in range(population_size):
        model = self.copy()
        model.alpha = rng.uniform(-np.pi, np.pi, model.alpha.shape)
        model.beta = rng.uniform(-np.pi, np.pi, model.beta.shape)
        population.append(model)

    num_iterations = int(np.log(final_temp / initial_temp) / np.log(cooling_rate))
    
    best_energies = []
    all_conformations = []
    
    temperatures = initial_temp * np.power(cooling_rate, np.arange(num_iterations))
    
    current_population = population 


    with multiprocessing.Pool(processes=min(population_size, multiprocessing.cpu_count())) as pool:

        with tqdm(
            total=num_iterations, 
            desc="Initializing", 
            unit="gen",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}"
        ) as pbar:
            
            for generation in range(num_iterations):
                current_temp = temperatures[generation]
                
                # Create tasks with unique seeds for reproducibility
                anneal_tasks = [
                    (model, current_temp, lam, chain_length, generation * population_size + i) 
                    for i, model in enumerate(current_population)
                ]
                
                # Parallel annealing
                mutated_population = pool.starmap(anneal_individual, anneal_tasks)
                # Calculate energies and track best
                mutated_energies = np.array([model.energy() for model in mutated_population])
                
                best_idx = np.argmin(mutated_energies)
                current_best = mutated_energies[best_idx]
                best_energies.append(current_best)
                all_conformations.append(mutated_population[best_idx].conformation.copy())
                
                avg_energy = np.mean(mutated_energies)
                energy_std = np.std(mutated_energies)
                
                pbar.set_description(
                    f"Gen {generation+1:3d} | T={current_temp:.2e} | "
                    f"Best: {current_best:7.2f} | Avg: {avg_energy:7.2f} | Std: {energy_std:6.2f}"
                )
                
                # Selection and crossover for next generation
                new_population = []
                
                # Elitism: keep best individual
                new_population.append(mutated_population[best_idx].copy())
                
                # Generate rest of population through tournament selection and crossover
                while len(new_population) < population_size:
                    tournament_size = 3
                    tournament_indices = rng.integers(0, population_size, tournament_size)
                    tournament_fitness = mutated_energies[tournament_indices]
                    
                    sorted_indices = np.argsort(tournament_fitness)
                    parent1_idx = tournament_indices[sorted_indices[0]]
                    parent2_idx = tournament_indices[sorted_indices[1]]
                    
                    parent1 = mutated_population[parent1_idx]
                    parent2 = mutated_population[parent2_idx]
                    
                    if rng.uniform() < crossover_rate:
                        child = parent1.breed(parent2)
                        new_population.append(child)
                    else:
                        new_population.append(parent1.copy())
                
                current_population = new_population
                
                pbar.update(1)
    
    final_energies = np.array([model.energy() for model in current_population])
    best_individual = current_population[np.argmin(final_energies)]
    
    return RunData(
        energies=np.array(best_energies),
        optimal_energy=np.min(best_energies),
        accepts=np.zeros(num_iterations),  # Not tracked in GA
        rejects=np.zeros(num_iterations),  # Not tracked in GA  
        temperatures=temperatures,
        conformations=all_conformations
    )

def anneal_individual(model, temp: float, lam: float, chain_length: int, seed: Optional[int] = None):
    
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    individual = model.copy()
    
    for k in range(chain_length):
        current_energy = individual.energy() 
        
        individual.propose(lam, ts=k/chain_length, rng=rng)
        new_energy = individual.energy()
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.uniform() < np.exp(-delta_e / temp):
            continue
        else:
            individual.revert()
    
    return individual
