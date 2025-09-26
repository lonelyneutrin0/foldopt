"""Irback's off-lattice protein model implementation."""
from typing import Optional, Self
from .base import ProteinModel

from numpy.typing import NDArray
import numpy as np

"""Descriptor for bond and torsion angles, with periodic wrapping to [-π, π]."""
class Angle:
    def __init__(self, size: int, alpha: Optional[NDArray] = None):
        if alpha is None: 
            self._value = -np.pi + 2*np.pi*np.random.uniform(size=(size,), low=0.0, high=1.0)
        
        else: 
            if len(alpha) != size:
                raise ValueError(f"Alpha array must be of size {size}")
            self._value = alpha
        
        self.size = size

    def __set__(self, obj, value):
        """
        Set bond angle values with automatic wrapping.
        
        Args:
            obj: The instance
            value: New angle values (array-like)
        """
        value = np.array(value)
        
        # Validate size
        if len(value) != self.size:
            raise ValueError(f"Alpha array must be of size {self.size}")
        
        # Apply wrapping and store
        self._value = self._wrap_angles(value)

    def __array__(self,) -> NDArray:
        """Return the angle values as a NumPy array."""
        return self._value

    def __get__(self, obj, objtype=None) -> NDArray:
        """Get the current bond angle values."""
        return self._value
    
    def _wrap_angles(self, angles: NDArray) -> NDArray:
        """Wrap angles to the range [-pi, pi]."""
        needs_wrapping = (angles > 2*np.pi) | (angles < -2*np.pi)
        
        if np.any(needs_wrapping):
            # Wrap to [-π, π] range using modulo operation
            # np.mod gives [0, 2π), so we shift to [-π, π)
            wrapped = np.mod(angles + np.pi, 2*np.pi) - np.pi
            return wrapped
        
        return angles
        
class IrbackModel(ProteinModel):
    """Irback's off-lattice protein model implementation."""

    residues: NDArray[np.bool]
    """Hydrophobicity of the residues. 1 for hydrophilic, 0 for hydrophobic."""


    def __init__(self, sequence: str, alpha: Optional[NDArray] = None, beta: Optional[NDArray] = None):
        self.sequence = sequence
        self.residues = self._map_residue_hydrophobicity(sequence)
        self.size = len(sequence)

        self._alpha = Angle(self.size-2, alpha)
        self._beta = Angle(self.size-3, beta)

    @property
    def alpha(self,) -> NDArray:
        """Get the current bond angle values."""
        return self._alpha._value
    
    @alpha.setter
    def alpha(self, value: NDArray):
        self._alpha.__set__(self, value)
    
    @property
    def beta(self,) -> NDArray:
        """Get the current torsion angle values."""
        return self._beta._value
    
    @beta.setter
    def beta(self, value: NDArray):
        self._beta.__set__(self, value)
    
    def _map_residue_hydrophobicity(self, residue: str) -> NDArray[np.bool]:
        """Map hydrophilic residues to 1, and hydrophobic residues to 0."""

        # This code could be a lot better, but works
        amino_acid_map = {'M': 'A', 'D': 'B', 'A': 'A', 'K': 'B', 'R': 'B', 'N': 'B', 'C': 'A', 'L': 'A', 'Q': 'B', 'H': 'B', 'E': 'B', 'I': 'A', 'T': 'B', 'S': 'B', 'Y': 'B', 'G': 'A', 'F': 'B', 'V': 'A', 'P': 'A', 'W': 'B'}
        hydrophobic = {k: v for k, v in amino_acid_map.items() if v == 'A'}

        return np.array([res in hydrophobic for res in residue], dtype=bool)

    def _get_coefficients(self,) -> NDArray:
        """Get the coefficients for the energy calculation."""

        residues = self.residues.astype(float) 
        coeff = residues[:, np.newaxis]*residues 
        coeff[coeff == 0] = 0.5
        return coeff 

    @property
    def conformation(self,) -> NDArray:
        """Get the current conformation as a 2D array of shape (N, 3)."""
        cos_alpha = np.cos(self.alpha)
        sin_alpha = np.sin(self.alpha)
        cos_beta = np.cos(self.beta)
        sin_beta = np.sin(self.beta)
        
        positions = np.zeros((self.size, 3), dtype=np.float64)
        positions[0] = np.array([0.0, 0.0, 0.0])
        positions[1] = np.array([0.0, 1.0, 0.0])
        positions[2] = positions[1] + np.array([cos_alpha[0], sin_alpha[0], 0])

        for i in range(self.size-3):
            positions[i+3] = positions[i+2] + \
                            np.array([cos_alpha[i+1]*cos_beta[i], sin_alpha[i+1]*cos_beta[i], sin_beta[i]])

        return positions

    def energy(self, conf: Optional[NDArray] = None) -> float:
        """Calculate the energy of the current conformation."""
        if conf is None:
            conf = self.conformation

        backbone_bending = np.sum(np.cos(self.alpha))
        torsion_energy = -0.5 * np.sum(np.cos(self.beta))

        distance_matrix = np.linalg.norm(conf[:, np.newaxis] - conf, axis=-1)
        np.fill_diagonal(distance_matrix, np.inf)
        distance_matrix = distance_matrix**-12 - distance_matrix**-6

        total_energy = backbone_bending + torsion_energy + np.sum(np.triu(4*distance_matrix*self._get_coefficients(), k=2))
        return total_energy
    
    def perturb(self, lam: float, ts: float, rng: np.random.Generator) -> Self:
        """Generate a new IrbackModel by perturbing the current one.

        Args:
            lam (float): Perturbation factor. Higher values lead to larger distances
                         in the conformation space between current and proposed states.
            ts (float): Timescale parameter between 0 and 1.
            rng (np.random.Generator, optional): Random number generator for reproducibility.
        
        Returns:
            IrbackModel: New perturbed IrbackModel instance.
        """
        random_i = np.random.randint(self.alpha.shape[0] + self.beta.shape[0])

        change = (np.random.uniform(0, 1) - 0.5)*np.random.uniform(0, 1) * (1 - ts)**lam

        if random_i < self.alpha.size:
            new_alpha = self.alpha.copy()
            new_alpha[random_i] += change
            new_beta = self.beta.copy() 
        
        else: 
            new_beta = self.beta.copy()
            new_beta[random_i - self.alpha.size] += change
            new_alpha = self.alpha.copy()

        return self.__class__(sequence=self.sequence, alpha=new_alpha, beta=new_beta)

    def mirror(self, other: Self):
        self.sequence = other.sequence
        self.residues = other.residues.copy()
        self.size = other.size
        self.alpha = other.alpha.copy()
        self.beta = other.beta.copy()

