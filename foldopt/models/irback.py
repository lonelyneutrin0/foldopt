"""Irback's off-lattice protein model implementation."""
from typing import Optional, Self
from .base import ProteinModel

from numpy.typing import NDArray
import numpy as np

from scipy.spatial import cKDTree

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

        if hasattr(obj, '_invalidate_caches'):
            obj._invalidate_caches()

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

    residues: NDArray
    """Hydrophobicity of the residues. 1 for hydrophilic, 0 for hydrophobic."""


    def __init__(self, sequence: str, alpha: Optional[NDArray] = None, beta: Optional[NDArray] = None, 
                 use_cutoff: bool = True, cutoff_distance: float = 10.0):
        self.sequence = sequence
        self.residues = self._map_residue_hydrophobicity(sequence)
        self.size = len(sequence)

        self.use_cutoff = use_cutoff
        self.cutoff_distance = cutoff_distance

        self._alpha = Angle(self.size-2, alpha)
        self._beta = Angle(self.size-3, beta)

        self._coefficients = self._get_coefficients()

        self._trig_cache_valid = False
        self._cos_alpha = None 
        self._sin_alpha = None
        self._cos_beta = None
        self._sin_beta = None

        self._conformation_cache_valid = False
        self._cached_conformation = None 

        self._energy_cache_valid = False
        self._cached_energy = None

        self._tree_cache_valid = False
        self._cached_tree = None
        self._tree_positions = None 

        self._positions_work = np.zeros((self.size, 3), dtype=np.float64)
        self._distance_matrix_work = np.zeros((self.size, self.size), dtype=np.float64)

        self._upper_triu_mask = np.triu(np.ones((self.size, self.size), dtype=bool), k=2)
        
        # Initialize state tracking for propose/revert
        self.perturb_idx = None
        self.change = None

        self.cache_stats = { 
            'conformation_hits': 0,
            'conformation_misses': 0,
            'energy_hits': 0,
            'energy_misses': 0,
            'trig_hits': 0,
            'trig_misses': 0,
            'tree_hits': 0,
            'tree_misses': 0,
        }

    def _invalidate_caches(self,):
        """Invalidate all cached values."""
        self._trig_cache_valid = False
        self._conformation_cache_valid = False
        self._energy_cache_valid = False
        self._tree_cache_valid = False
    
    def _update_trig_cache(self,):
        """Update the cached trigonometric values if invalid."""
        if not self._trig_cache_valid:
            self._cos_alpha = np.cos(self.alpha)
            self._sin_alpha = np.sin(self.alpha)
            self._cos_beta = np.cos(self.beta)
            self._sin_beta = np.sin(self.beta)

            self._trig_cache_valid = True
            self.cache_stats['trig_misses'] += 1
        else:
            self.cache_stats['trig_hits'] += 1

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
    
    def _map_residue_hydrophobicity(self, residue: str) -> NDArray:
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

        if self._conformation_cache_valid and self._cached_conformation is not None:
            self.cache_stats['conformation_hits'] += 1
            return self._cached_conformation

        self.cache_stats['conformation_misses'] += 1
        self._update_conformation()
        return self._cached_conformation
    
    def _update_conformation(self,):
        """Update the cached conformation if invalid."""

        self._update_trig_cache()

        positions = self._positions_work
        positions.fill(0.0)
        
        positions[0] = np.array([0.0, 0.0, 0.0])
        positions[1] = np.array([0.0, 1.0, 0.0])
        positions[2] = positions[1] + np.array([self._cos_alpha[0], self._sin_alpha[0], 0])

        for i in range(self.size-3):
            positions[i+3] = positions[i+2] + \
                            np.array([self._cos_alpha[i+1]*self._cos_beta[i], self._sin_alpha[i+1]*self._cos_beta[i], self._sin_beta[i]])

        self._cached_conformation = positions.copy()
        self._conformation_cache_valid = True

    def energy(self, conf: Optional[NDArray] = None) -> float:
        """Calculate the energy of the current conformation."""
        if conf is None and self._energy_cache_valid and self._cached_energy is not None:
            self.cache_stats['energy_hits'] += 1
            return self._cached_energy
        
        self.cache_stats['energy_misses'] += 1

        if conf is None:
            conf = self.conformation
        
        self._update_trig_cache()

        backbone_bending = np.sum(self._cos_alpha)
        torsion_energy = -0.5 * np.sum(self._cos_beta)

        if self.use_cutoff:
            interaction_energy = self._compute_sparse_interaction_energy(conf)
        else:
            interaction_energy = self._compute_dense_interaction_energy(conf)

        total_energy = backbone_bending + torsion_energy + interaction_energy

        # Cache the result if using current conformation
        if conf is None:
            self._cached_energy = total_energy
            self._energy_cache_valid = True

        return total_energy
    
    def _compute_sparse_interaction_energy(self, conf: NDArray) -> float:
        """Compute interaction energy using a cutoff distance for efficiency."""
        self._build_tree(conf)

        pairs = self._cached_tree.query_pairs(self.cutoff_distance, output_type='ndarray')

        if len(pairs) == 0:
            return 0.0
        
        valid_pairs = pairs[np.abs(pairs[:,0] - pairs[:,1]) >= 2]

        if len(valid_pairs) == 0:
            return 0.0
        
        i_indices = valid_pairs[:,0]
        j_indices = valid_pairs[:,1]

        distances = np.linalg.norm(conf[i_indices] - conf[j_indices], axis=1)

        inv_r6 = distances**-6
        inv_r12 = inv_r6 * inv_r6

        lj = (inv_r12 - inv_r6)

        coeffs = self._coefficients[i_indices, j_indices]
        interaction_energy = np.sum(4 * lj * coeffs)

        return interaction_energy

    def _compute_dense_interaction_energy(self, conf: NDArray) -> float:
        """Compute interaction energy using the full distance matrix."""
        distance_matrix = np.linalg.norm(conf[:, np.newaxis] - conf, axis=-1)

        inv_r6 = distance_matrix**-6
        inv_r12 = inv_r6 * inv_r6
        lj = (inv_r12 - inv_r6)

        interaction_energy = np.sum(4 * lj[self._upper_triu_mask] * self._coefficients[self._upper_triu_mask])

        return interaction_energy

    def _build_tree(self, conf: NDArray):
        """Build a spatial tree for efficient neighbor searching."""
        if (not self._tree_cache_valid or 
            self._cached_tree is None or
            self._tree_positions is None or
            not np.array_equal(self._tree_positions, conf)):

            self._cached_tree = cKDTree(conf)
            self._tree_positions = conf.copy()
            self._tree_cache_valid = True
            self.cache_stats['tree_misses'] += 1
        else:
            self.cache_stats['tree_hits'] += 1
    
    def propose(self, lam: float, ts: float, rng: np.random.Generator):
        """Generate a new proposed state by perturbing either a bond or torsion angle.

        Args:
            lam (float): Perturbation factor. Higher values lead to larger distances
                         in the conformation space between current and proposed states.
            ts (float): Timescale parameter between 0 and 1.
            rng (np.random.Generator, optional): Random number generator for reproducibility.
        
        """
        if rng is None:
            rng = np.random.default_rng()
            
        random_i = rng.integers(0, self.alpha.shape[0] + self.beta.shape[0])

        change = (rng.uniform(0, 1) - 0.5) * rng.uniform(0, 1) * (1 - ts)**lam

        if random_i < self.alpha.size:
            self.alpha[random_i] += change

        else:
            self.beta[random_i - self.alpha.size] += change

        # Manually invalidate caches since we modified angles directly
        self._invalidate_caches()
        
        self.perturb_idx = random_i
        self.change = change

    def revert(self, ): 
        """Revert the last proposed change."""
        if self.perturb_idx is None:
            return 
        
        if self.perturb_idx < self.alpha.size:
            self.alpha[self.perturb_idx] -= self.change

        else:
            self.beta[self.perturb_idx - self.alpha.size] -= self.change
            
        # Manually invalidate caches since we modified angles directly
        self._invalidate_caches()

    def copy(self,):
        """Create a deep copy of the current model instance."""
        new_model = IrbackModel(self.sequence, self.alpha.copy(), self.beta.copy(), 
                                use_cutoff=self.use_cutoff, cutoff_distance=self.cutoff_distance)
        
        new_model.perturb_idx = self.perturb_idx
        new_model.change = self.change

        new_model.cache_stats = self.cache_stats.copy()
        
        # Copy caches if valid
        if self._conformation_cache_valid:
            new_model._cached_conformation = self._cached_conformation.copy()
            new_model._conformation_cache_valid = True
        
        if self._energy_cache_valid:
            new_model._cached_energy = self._cached_energy
            new_model._energy_cache_valid = True
        
        if self._trig_cache_valid:
            new_model._cos_alpha = self._cos_alpha.copy()
            new_model._sin_alpha = self._sin_alpha.copy()
            new_model._cos_beta = self._cos_beta.copy()
            new_model._sin_beta = self._sin_beta.copy()
            new_model._trig_cache_valid = True
        
        if self._tree_cache_valid and self._cached_tree is not None and self._tree_positions is not None:
            new_model._cached_tree = cKDTree(self._tree_positions.copy())
            new_model._tree_positions = self._tree_positions.copy()
            new_model._tree_cache_valid = True

        return new_model
    
    def breed(self, other: Self) -> Self:
        """Breed with another Irback model to produce an offspring.

        Args:
            other (IrbackModel): Another IrbackModel instance to breed with.

        Returns:
            IrbackModel: A new IrbackModel instance representing the offspring.
        """
        if not isinstance(other, self.__class__):
            raise ValueError("Can only breed with another IrbackModel instance.")
        
        if self.size != other.size:
            raise ValueError("Both models must have the same sequence length to breed.")

        # Single-point crossover for alpha and beta angles
        alpha_crossover = np.random.randint(1, self.alpha.size)
        beta_crossover = np.random.randint(1, self.beta.size)

        child_alpha = np.concatenate((self.alpha[:alpha_crossover], other.alpha[alpha_crossover:]))
        child_beta = np.concatenate((self.beta[:beta_crossover], other.beta[beta_crossover:]))

        new_model = self.__class__(self.sequence, child_alpha, child_beta, 
                                   use_cutoff=self.use_cutoff, cutoff_distance=self.cutoff_distance)
        
        return new_model
