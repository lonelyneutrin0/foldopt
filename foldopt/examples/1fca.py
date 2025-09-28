from foldopt.models import IrbackModel
from foldopt.opt import OptAlgo

import matplotlib.pyplot as plt
import numpy as np
import cProfile
import pstats
from pstats import SortKey

def run_optimization():
    """Function to wrap the optimization for profiling."""
    x = IrbackModel(sequence="AYVINEACISCGACEPECPVDAISQGGSRYVIDADTCIDCGACAGVCPVDAPVQA")
    x.set_optimizer(OptAlgo.METROPOLIS_ANNEALING)
    out = x.optimize(initial_temp=1.0, final_temp=1e-12, cooling_rate=0.99, lam = 3.0, chain_length=10_000)
    
    print(f'Optimal energy: {out.optimal_energy}')
    print(out.conformations[0].shape)
    np.savez('1fca_run.npz', energies=out.energies, accepts=out.accepts, rejects=out.rejects, 
             conformations=out.conformations)
    
    return out

if __name__ == "__main__":
    # Profile the optimization
    print("Starting profiled optimization...")
    profiler = cProfile.Profile()
    profiler.enable()

    out = run_optimization()

    profiler.disable()

    # Save and analyze profile results  
    profiler.dump_stats('metropolis_profile.prof')

    # Print top functions by cumulative time
    print("\n" + "="*80)
    print("TOP FUNCTIONS BY CUMULATIVE TIME:")
    print("="*80)
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Top 20 functions

    print("\n" + "="*80)
    print("TOP FUNCTIONS BY TOTAL TIME:")
    print("="*80)
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(20)  # Top 20 functions

    print("\n" + "="*80)
    print("TOP FUNCTIONS BY CALL COUNT:")
    print("="*80)
    stats.sort_stats(SortKey.CALLS)
    stats.print_stats(20)  # Top 20 functions

    # Plot the energy curve
    plt.figure(figsize=(10, 6))
    plt.plot(out.energies)
    plt.title('Energy vs Iteration (after 1000 iterations)')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.show()
    plt.close()

    print(f"\nProfile data saved to 'metropolis_profile.prof'")
    print("You can load it later with: python -m pstats metropolis_profile.prof")
    
    # Additional analysis: Print method-specific bottlenecks
    print("\n" + "="*80)
    print("ANALYSIS: Looking for specific bottlenecks")
    print("="*80)
    
    # Filter for specific methods that are likely bottlenecks
    print("\nEnergy calculation methods:")
    stats.print_stats("energy")
    
    print("\nConformation methods:")
    stats.print_stats("conformation")
    
    print("\nNumPy operations:")
    stats.print_stats("numpy")
    
    print("\nMatrix operations:")
    stats.print_stats("linalg|norm|triu")