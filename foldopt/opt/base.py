"""Base classes and methods for optimization algorithms."""
from enum import Enum

from .metropolis_annealing import metropolis_annealing
from .genetic_annealing import genetic_annealing

class OptAlgo(Enum):
    """Enumeration of optimization algorithms."""
    METROPOLIS_ANNEALING = "metropolis"
    GENETIC_ANNEALING  = "genetic"

    @property
    def value(self): 
        if self == OptAlgo.METROPOLIS_ANNEALING:
            return metropolis_annealing
        elif self == OptAlgo.GENETIC_ANNEALING:
            return genetic_annealing
        else:
            raise ValueError(f"Unknown optimization algorithm: {self}")