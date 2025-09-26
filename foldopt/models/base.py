"""Abstract class for protein models (Like Irback, HP, etc)"""
from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from typing import Self

from ..opt.base import OptAlgo

class ProteinModel(ABC):
    """Abstract base class for protein models."""

    def __init__(self,): 
        pass

    @abstractmethod
    def energy(self) -> float:
        """Calculate the energy of the current conformation."""
        pass
    
    @property
    @abstractmethod
    def conformation(self,) -> ArrayLike:
        pass 

    @abstractmethod
    def perturb(self, *args, **kwargs) -> Self:
        """Generate a new ProteinModel by perturbing the current one.

        Args:
            lam (float): Perturbation factor. Higher values lead to larger distances
                         in the conformation space between current and proposed states.
            rng (np.random.Generator, optional): Random number generator for reproducibility.
        """
        pass

    @abstractmethod
    def mirror(self, *args, **kwargs):
        """Mirror the parameters of a different protein model."""
        pass
    
    def set_optimizer(self, opt: OptAlgo): 
        """Set the optimizer for the protein model."""
        self.optimizer = opt.value

    def optimize(self, *args, **kwargs): 
        return self.optimizer(self, *args, **kwargs)
