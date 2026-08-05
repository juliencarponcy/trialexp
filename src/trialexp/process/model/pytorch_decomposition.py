#this is a test
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import normalize
from sklearn.decomposition import sparse_encode
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
from loguru import logger

class SymmetricSigmoid(nn.Module):
    def __init__(self, c=0.5, k=10.0, scale=1.0, shift=0.0):
        """
        c: half-distance of the sigmoid midpoints (use 0.5 from your idea)
        k: sharpness (higher -> steeper transitions)
        scale: multiply output
        shift: add baseline
        """
        super().__init__()
        self.c = c
        self.k = k
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        s_pos = torch.sigmoid(self.k * (x - self.c))
        s_neg = torch.sigmoid(-self.k * (x + self.c))
        y = s_pos - s_neg
        # y in (-1,1) roughly; optionally scale & clamp
        return (y * self.scale + self.shift).clamp(-1.0, 1.0)

class TwoExpFastEnd(nn.Module):
    """
    Two-sided saturating exponentials that stay near zero for most of [-1,1]
    and then rise rapidly near ±1.
    y(x) =  A_pos * (1 - exp(-k_pos * (x_pos**alpha)))   for x >= 0
          -A_neg * (1 - exp(-k_neg * (x_neg**alpha)))   for x <  0
    """
    def __init__(self, alpha=10, A_pos=1e-2, A_neg=1e-2):
        super().__init__()
        self.alpha = float(alpha)
        self.A_pos = float(A_pos)
        self.A_neg = float(A_neg)

    def forward(self, x):
        # assume x in [-1,1]; if not, you can clamp or scale beforehand
        x_pos = x.clamp(min=0.0)
        x_neg = (-x).clamp(min=0.0)

        # raise to power alpha -> delays rise if alpha > 1

        y_pos = self.A_pos * torch.exp(x_pos*self.alpha)
        y_neg = -self.A_neg * torch.exp(x_neg*self.alpha)
        y = (y_pos+y_neg)
        return y.clamp(-1,1)
    
def naka_rushton(c, R_max=1.0, c50=1.0, n=2.0):
    # Naka-Rushton function
    num = c**n
    den = num + c50**n
    return R_max * (num / (den + 1e-12))

def sigmoid(c, R_max=1.0, c50=1.0, slope=1.0):
    return R_max / (1 + np.exp(-slope * (c - c50)))

class LearnableSigmoid(nn.Module):
    """
    Learnable sigmoid activation function with trainable parameters.
    
    This module implements a parameterized sigmoid function where the maximum response (Rmax),
    slope, and midpoint (c50) are learnable parameters that can be optimized during training.
    
    The sigmoid function is defined as:
        f(x) = Rmax / (1 + exp(-slope * (x - c50)))
    
    where:
        - Rmax controls the maximum output value
        - slope controls the steepness of the sigmoid curve
        - c50 is the midpoint (the input value where output is Rmax/2)
    
    Parameters
    ----------
    init_Rmax : float, optional
        Initial value for the maximum response parameter. Default is 1.0.
    init_slope : float, optional
        Initial value for the slope parameter. Default is 4.
    init_c50 : float, optional
        Initial value for the midpoint parameter. Default is 1.
    separate_pos_neg : bool, optional
        If True, use separate parameters for positive and negative inputs. Default is True.
    single_sided : bool, optional
        If True, only process positive inputs (assumes x >= 0). Default is False.
        When True, separate_pos_neg is ignored.
    
    Attributes
    ----------
    Rmax_pos : nn.Parameter
        Learnable parameter controlling the maximum output value for positive inputs.
    slope_pos : nn.Parameter
        Learnable parameter controlling the steepness of the sigmoid for positive inputs.
    c50_pos : nn.Parameter
        Learnable parameter controlling the midpoint of the sigmoid for positive inputs.
    Rmax_neg : nn.Parameter
        Learnable parameter controlling the maximum output value for negative inputs (if separate_pos_neg=True).
    slope_neg : nn.Parameter
        Learnable parameter controlling the steepness of the sigmoid for negative inputs (if separate_pos_neg=True).
    c50_neg : nn.Parameter
        Learnable parameter controlling the midpoint of the sigmoid for negative inputs (if separate_pos_neg=True).
    """

    def __init__(self, init_Rmax=1.0, init_slope=4.0, init_c50=1.0, separate_pos_neg=True, single_sided=False):
        super().__init__()
        # Learnable parameters for positive values
        self.Rmax_pos = nn.Parameter(torch.tensor(init_Rmax))
        # self.Rmax_pos = torch.tensor(init_Rmax)

        self.slope_pos = nn.Parameter(torch.tensor(init_slope))
        self.c50_pos = nn.Parameter(torch.tensor(init_c50))
        
        self.single_sided = single_sided
        self.separate_pos_neg = separate_pos_neg and not single_sided  # Override if single_sided
        
        # Always initialize separate parameters for negative values
        self.Rmax_neg = nn.Parameter(torch.tensor(init_Rmax))
        # self.Rmax_neg = torch.tensor(init_Rmax)
        self.slope_neg = nn.Parameter(torch.tensor(init_slope))
        self.c50_neg = nn.Parameter(torch.tensor(init_c50))

    def forward(self, x):
        """
        Apply a sigmoidal activation function to the input.
        
        This function implements a sigmoid transformation of the form:
        R_max / (1 + exp(-slope * (x - c50)))
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
                                For single_sided=True, assumes x >= 0
        
        Returns:
            torch.Tensor: Transformed tensor with the same shape as input.
        """
        if self.single_sided:
            # Fast path: only positive sigmoid, no clamping or negative processing
            # Assumes x >= 0 (e.g., firing rates in per_neuron mode)
            return self.Rmax_pos / (1 + torch.exp(-self.slope_pos * (x - self.c50_pos)))
        
        # Two-sided implementation with separate parameters for positive and negative
        x_pos = x.clamp(min=0.0)
        x_neg = (-x).clamp(min=0.0)
        
        # Sigmoid for positive values
        y_pos = self.Rmax_pos / (1 + torch.exp(-self.slope_pos * (x_pos - self.c50_pos)))
        
        # Always use separate parameters for negative values
        y_neg = -self.Rmax_neg / (1 + torch.exp(-self.slope_neg * (x_neg - self.c50_neg)))
        
        return y_pos + y_neg


class LearnableExponential(nn.Module):
    """
    Learnable exponential activation for modeling DA burst release.

    Models the relationship where high firing rates (burst mode) lead to
    exponential increase in DA release.

    For positive values: y = scale * (exp(rate * x) - 1)
    For negative values: y = -scale * (exp(rate * |x|) - 1)

    The rate and scale parameters are learned during optimization.
    Scale parameters are constrained to be positive to prevent sign-flipping.

    Parameters:
    -----------
    init_rate : float
        Initial exponential rate parameter
    init_scale : float
        Initial scale parameter (will be transformed to ensure positivity)
    separate_pos_neg : bool
        If True, use separate parameters for positive and negative inputs
    single_sided : bool
        If True, only process positive inputs (assumes x >= 0).
        This mode is faster for per-neuron activation where firing rates are always positive.
        When True, separate_pos_neg is ignored.

    TODO: use the Naka-Ruston activation function
    """
    def __init__(self, init_rate=1.0, init_scale=1.0, separate_pos_neg=True, single_sided=False):
        super().__init__()
        # Learnable parameters for exponential rate and scale
        # Store unconstrained parameters; will apply softplus in forward pass to ensure scale > 0
        self.rate_pos = nn.Parameter(torch.tensor(init_rate))
        self.scale_pos_raw = nn.Parameter(torch.tensor(self._inverse_softplus(init_scale)))

        self.single_sided = single_sided
        self.separate_pos_neg = separate_pos_neg and not single_sided  # Override if single_sided

        if self.separate_pos_neg:
            self.rate_neg = nn.Parameter(torch.tensor(init_rate))
            self.scale_neg_raw = nn.Parameter(torch.tensor(self._inverse_softplus(init_scale)))

    @staticmethod
    def _inverse_softplus(y):
        """Inverse of softplus: x = log(exp(y) - 1)"""
        return float(torch.log(torch.exp(torch.tensor(y)) - 1).item())

    @property
    def scale_pos(self):
        """Positive-constrained scale parameter"""
        return F.softplus(self.scale_pos_raw)

    @property
    def scale_neg(self):
        """Positive-constrained scale parameter for negative branch"""
        if self.separate_pos_neg:
            return F.softplus(self.scale_neg_raw)
        return self.scale_pos

    def forward(self, x):
        """
        Forward pass supporting both single and batched inputs.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (..., n_features) where ... can be any number of batch dimensions
            For single_sided=True, assumes x >= 0

        Returns:
        --------
        torch.Tensor
            Output tensor of same shape as input
        """
        if self.single_sided:
            # Fast path: only positive exponential, no clamping or negative processing
            # Assumes x >= 0 (e.g., firing rates in per_neuron mode)
            return self.scale_pos * (torch.exp(self.rate_pos * x) - 1)

        # Original two-sided implementation
        x_pos = x.clamp(min=0.0)
        x_neg = (-x).clamp(min=0.0)

        # Exponential for positive (burst mode)
        # Using exp(rate*x) - 1 ensures y=0 when x=0
        y_pos = self.scale_pos * (torch.exp(self.rate_pos * x_pos) - 1)

        if self.separate_pos_neg:
            y_neg = -self.scale_neg * (torch.exp(self.rate_neg * x_neg) - 1)
        else:
            y_neg = -self.scale_pos * (torch.exp(self.rate_pos * x_neg) - 1)

        return y_pos + y_neg

class SigmoidSparseEncoder(nn.Module):
    """
    Sparse encoder with learnable activation transformation for linear mapping.

    The idea is: target = dictionary @ activation(code)
    This allows the linear combination to work with learnable nonlinear transformations.

    Parameters:
    -----------
    dictionary : torch.Tensor
        Dictionary of atoms, shape (n_components, n_features)
    activation_type : str
        Type of activation: 'global' (single activation after sum),
        'per_neuron' (separate activation per neuron), or 'legacy' (old TwoExpFastEnd)
        Default: 'global'
    activation_function : str
        Type of activation function: 'exponential' or 'sigmoid'
        Only used when activation_type is 'global' or 'per_neuron'
        Default: 'sigmoid'
    """

    def __init__(self, dictionary: torch.Tensor, activation_type='global', activation_function='sigmoid'):
        super().__init__()
        # self.n_components = n_components
        self.dictionary = dictionary
        self.n_components = self.dictionary.shape[0]
        # Learnable code parameters (before sigmoid)
        self.code = nn.Parameter(torch.randn(1, self.n_components) * 0.01)
        self.activation_type = activation_type
        self.activation_function = activation_function

        if activation_type == 'global':
            # Single activation applied after summation
            if activation_function == 'exponential':
                self.act = LearnableExponential(init_rate=1.0, init_scale=0.1)
            elif activation_function == 'sigmoid':
                self.act = LearnableSigmoid(init_Rmax=1.0, init_slope=4.0, init_c50=1.0)
            elif activation_function == 'tanh':
                self.act = nn.Tanh()
            else:
                raise ValueError(f"Unknown activation_function: {activation_function}")
        elif activation_type == 'per_neuron':
            # Separate activation for each neuron/dictionary element
            # Use single_sided=True since firing rates are always positive
            if activation_function == 'exponential':
                self.activations = nn.ModuleList([
                    LearnableExponential(init_rate=1.0, init_scale=0.1, single_sided=True)
                    for _ in range(self.n_components)
                ])
            elif activation_function == 'sigmoid':
                self.activations = nn.ModuleList([
                    LearnableSigmoid(init_Rmax=1.0, init_slope=8.0, init_c50=1.0, single_sided=True)
                    for _ in range(self.n_components)
                ])
            elif activation_function == 'tanh':
                self.activations = nn.ModuleList([
                    nn.Tanh()
                    for _ in range(self.n_components)
                ])
            else:
                raise ValueError(f"Unknown activation_function: {activation_function}")
        elif activation_type == 'legacy':
            # Old TwoExpFastEnd for compatibility
            self.act = TwoExpFastEnd()
        else:
            raise ValueError(f"Unknown activation_type: {activation_type}")

    def forward(self) -> torch.Tensor:
        """
        Forward pass: reconstruct signal from dictionary.

        Returns:
        --------
        reconstruction : torch.Tensor
            Reconstructed signal, shape (1, n_features)
        """
        if self.activation_type == 'global' or self.activation_type == 'legacy':
            reconstruction = self.act(self.code @ self.dictionary)
            # reconstruction = self.code @ self.act(self.dictionary)


        elif self.activation_type == 'per_neuron':
            # FULLY VECTORIZED: Process all neurons in parallel
            # Apply activation to dictionary first
            # Shape: (n_components, n_features)
            # Multiply code with dictionary first to get neuron contributions
            # Apply per-neuron activation to dictionary first
            # Shape: (n_components, n_features)
            activated_dictionary = self._batch_activate_per_neuron(self.dictionary)
            
            # Then multiply by code
            # code shape: (1, n_components), activated_dictionary shape: (n_components, n_features)
            reconstruction = self.code @ activated_dictionary  # Shape: (1, n_features)

        return reconstruction

    def _batch_activate_per_neuron(self, neuron_contributions: torch.Tensor) -> torch.Tensor:
        """
        Efficiently apply per-neuron activations in batch.

        This method dispatches to the appropriate batch activation function
        based on the type of activation module being used.

        Parameters:
        -----------
        neuron_contributions : torch.Tensor
            Shape (n_components, n_features) - contributions from each neuron

        Returns:
        --------
        torch.Tensor
            Shape (n_components, n_features) - activated contributions
        """
        # Dispatch to appropriate batch function based on activation type
        if isinstance(self.activations[0], LearnableExponential):
            return self._batch_activate_per_neuron_exponential(neuron_contributions)
        elif isinstance(self.activations[0], LearnableSigmoid):
            return self._batch_activate_per_neuron_sigmoid(neuron_contributions)
        else:
            raise NotImplementedError(f"Batch activation not implemented for {type(self.activations[0])}")

    def _batch_activate_per_neuron_exponential(self, neuron_contributions: torch.Tensor) -> torch.Tensor:
        """
        Efficiently apply per-neuron exponential activations in batch.

        This method extracts parameters from all LearnableExponential activation modules
        and applies them in a vectorized manner, which is much faster than calling each
        activation separately in a loop.

        Parameters:
        -----------
        neuron_contributions : torch.Tensor
            Shape (n_components, n_features) - contributions from each neuron

        Returns:
        --------
        torch.Tensor
            Shape (n_components, n_features) - activated contributions
        """
        # Extract all parameters as tensors
        # Shape: (n_components,)
        rates_pos = torch.stack([act.rate_pos for act in self.activations])
        scales_pos = torch.stack([act.scale_pos for act in self.activations])

        # Check if using single-sided activation (faster for per_neuron with positive firing rates)
        if self.activations[0].single_sided:
            # Fast path: only positive exponential, assumes neuron_contributions >= 0
            # Broadcast parameters: (n_components, 1) * (n_components, n_features)
            return scales_pos.unsqueeze(-1) * (torch.exp(rates_pos.unsqueeze(-1) * neuron_contributions) - 1)

        # Original two-sided implementation
        if self.activations[0].separate_pos_neg:
            rates_neg = torch.stack([act.rate_neg for act in self.activations])
            scales_neg = torch.stack([act.scale_neg for act in self.activations])

        # Apply activation in batch
        # Shape: (n_components, n_features)
        x_pos = neuron_contributions.clamp(min=0.0)
        x_neg = (-neuron_contributions).clamp(min=0.0)

        # Broadcast parameters: (n_components, 1) * (n_components, n_features)
        y_pos = scales_pos.unsqueeze(-1) * (torch.exp(rates_pos.unsqueeze(-1) * x_pos) - 1)

        if self.activations[0].separate_pos_neg:
            y_neg = -scales_neg.unsqueeze(-1) * (torch.exp(rates_neg.unsqueeze(-1) * x_neg) - 1)
        else:
            y_neg = -scales_pos.unsqueeze(-1) * (torch.exp(rates_pos.unsqueeze(-1) * x_neg) - 1)

        return y_pos + y_neg

    def _batch_activate_per_neuron_sigmoid(self, neuron_contributions: torch.Tensor) -> torch.Tensor:
        """
        Efficiently apply per-neuron sigmoid activations in batch.

        This method extracts parameters from all LearnableSigmoid activation modules
        and applies them in a vectorized manner, which is much faster than calling each
        activation separately in a loop.

        Parameters:
        -----------
        neuron_contributions : torch.Tensor
            Shape (n_components, n_features) - contributions from each neuron

        Returns:
        --------
        torch.Tensor
            Shape (n_components, n_features) - activated contributions
        """
        # Extract all parameters as tensors
        # Shape: (n_components,)
        Rmax_pos = torch.stack([act.Rmax_pos for act in self.activations])
        slope_pos = torch.stack([act.slope_pos for act in self.activations])
        c50_pos = torch.stack([act.c50_pos for act in self.activations])

        # Check if using single-sided activation (faster for per_neuron with positive firing rates)
        if self.activations[0].single_sided:
            # Fast path: only positive sigmoid, assumes neuron_contributions >= 0
            # Broadcast parameters: (n_components, 1) for all parameters
            # Formula: Rmax / (1 + exp(-slope * (x - c50)))
            return Rmax_pos.unsqueeze(-1) / (
                1 + torch.exp(-slope_pos.unsqueeze(-1) * (neuron_contributions - c50_pos.unsqueeze(-1)))
            )

        # Two-sided implementation with separate parameters for positive and negative
        # Extract negative parameters
        Rmax_neg = torch.stack([act.Rmax_neg for act in self.activations])
        slope_neg = torch.stack([act.slope_neg for act in self.activations])
        c50_neg = torch.stack([act.c50_neg for act in self.activations])

        # Apply activation in batch
        # Shape: (n_components, n_features)
        x_pos = neuron_contributions.clamp(min=0.0)
        x_neg = (-neuron_contributions).clamp(min=0.0)

        # Broadcast parameters: (n_components, 1) * (n_components, n_features)
        # Sigmoid for positive values: Rmax / (1 + exp(-slope * (x - c50)))
        y_pos = Rmax_pos.unsqueeze(-1) / (
            1 + torch.exp(-slope_pos.unsqueeze(-1) * (x_pos - c50_pos.unsqueeze(-1)))
        )

        # Sigmoid for negative values (note the negative sign on the output)
        y_neg = -Rmax_neg.unsqueeze(-1) / (
            1 + torch.exp(-slope_neg.unsqueeze(-1) * (x_neg - c50_neg.unsqueeze(-1)))
        )

        return y_pos + y_neg

    def get_code(self) -> torch.Tensor:
        """Get the raw code."""
        return self.code

    def get_activation_params(self) -> dict:
        """Get learned activation parameters for analysis."""
        if self.activation_type == 'per_neuron':
            # Determine activation type and extract appropriate parameters
            if isinstance(self.activations[0], LearnableExponential):
                params = {
                    'rates_pos': [act.rate_pos.item() for act in self.activations],
                    'scales_pos': [act.scale_pos.item() for act in self.activations],
                }
                if self.activations[0].separate_pos_neg:
                    params['rates_neg'] = [act.rate_neg.item() for act in self.activations]
                    params['scales_neg'] = [act.scale_neg.item() for act in self.activations]
            elif isinstance(self.activations[0], LearnableSigmoid):
                params = {
                    'Rmax_pos': [act.Rmax_pos.item() for act in self.activations],
                    'slope_pos': [act.slope_pos.item() for act in self.activations],
                    'c50_pos': [act.c50_pos.item() for act in self.activations],
                }
                if self.activations[0].separate_pos_neg:
                    params['Rmax_neg'] = [act.Rmax_neg.item() for act in self.activations]
                    params['slope_neg'] = [act.slope_neg.item() for act in self.activations]
                    params['c50_neg'] = [act.c50_neg.item() for act in self.activations]
            else:
                params = {}
        elif self.activation_type == 'global':
            # Determine activation type and extract appropriate parameters
            if isinstance(self.act, LearnableExponential):
                params = {
                    'rate_pos': self.act.rate_pos.item(),
                    'scale_pos': self.act.scale_pos.item(),
                }
                if self.act.separate_pos_neg:
                    params['rate_neg'] = self.act.rate_neg.item()
                    params['scale_neg'] = self.act.scale_neg.item()
            elif isinstance(self.act, LearnableSigmoid):
                params = {
                    'Rmax_pos': self.act.Rmax_pos.item(),
                    'slope_pos': self.act.slope_pos.item(),
                    'c50_pos': self.act.c50_pos.item(),
                }
                if self.act.separate_pos_neg:
                    params['Rmax_neg'] = self.act.Rmax_neg.item()
                    params['slope_neg'] = self.act.slope_neg.item()
                    params['c50_neg'] = self.act.c50_neg.item()
            else:
                params = {}
        else:
            params = {}
        return params

    def freeze_low_code_neurons(self, threshold: float = 0.01) -> int:
        """
        Freeze activation parameters for neurons with low code values.

        Parameters:
        -----------
        threshold : float
            Absolute code value threshold. Neurons with |code| < threshold
            will have their activation parameters frozen.

        Returns:
        --------
        n_frozen : int
            Number of neurons that were frozen
        """
        if self.activation_type != 'per_neuron':
            return 0  # Only applicable for per_neuron activation

        code_values = self.code.squeeze(0).abs()
        n_frozen = 0

        for i, code_val in enumerate(code_values):
            if code_val < threshold:
                # Freeze all parameters of this activation function
                for param in self.activations[i].parameters():
                    param.requires_grad = False
                n_frozen += 1

        return n_frozen

    def unfreeze_all_neurons(self):
        """Unfreeze all activation parameters."""
        if self.activation_type != 'per_neuron':
            return

        for activation in self.activations:
            for param in activation.parameters():
                param.requires_grad = True

def sparse_encode_pytorch_batch(
    targets: np.ndarray,
    dictionary: np.ndarray,
    sparsity_weight: float = 0.001,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    batch_size: int = 32,
    device: Optional[str] = None,
    verbose: bool = True,
    sparsity_type: str = 'L1',
    l1_ratio: float = 0.5
) -> Tuple[np.ndarray, dict]:
    """
    PyTorch-based sparse encoding for multiple targets (batch processing).

    Parameters:
    -----------
    targets : np.ndarray
        Target signals to decompose, shape (n_samples, n_features)
    dictionary : np.ndarray
        Dictionary of atoms, shape (n_components, n_features)
    sparsity_weight : float
        Regularization weight for sparsity (default: 0.001)
    learning_rate : float
        Learning rate for optimization
    n_iterations : int
        Maximum number of optimization iterations
    batch_size : int
        Batch size for processing
    device : str, optional
        Device to use ('cuda' or 'cpu')
    verbose : bool
        Whether to print progress
    sparsity_type : str
        Type of sparsity regularization: 'L1', 'L2', or 'elastic_net' (default: 'L1')
    l1_ratio : float
        Mixing parameter for elastic net (only used when sparsity_type='elastic_net')
        Range [0, 1] where 1.0 is pure L1 and 0.0 is pure L2 (default: 0.5)

    Returns:
    --------
    codes : np.ndarray
        Sparse codes after sigmoid transformation, shape (n_samples, n_components)
    info : dict
        Dictionary containing aggregated information
    """
    n_samples = targets.shape[0]
    n_components = dictionary.shape[0]

    codes_all = np.zeros((n_samples, n_components))
    loss_history_all = []

    # Process in batches
    n_batches = int(np.ceil(n_samples / batch_size))

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)

        if verbose:
            print(f"\nProcessing batch {batch_idx + 1}/{n_batches} "
                  f"(samples {start_idx} to {end_idx})")

        # Process each sample in the batch
        for i in range(start_idx, end_idx):
            code, info, reconstruction = sparse_encode_pytorch(
                targets[i:i+1],
                dictionary,
                sparsity_weight=sparsity_weight,
                learning_rate=learning_rate,
                n_iterations=n_iterations,
                device=device,
                verbose=False,
                sparsity_type=sparsity_type,
                l1_ratio=l1_ratio
            )

            codes_all[i] = code[0]
            loss_history_all.append(info['loss_history'])

        if verbose:
            avg_loss = np.mean([info['reconstruction_loss'] for info in [info]])
            print(f"Batch {batch_idx + 1} average reconstruction loss: {avg_loss:.6f}")

    info_all = {
        'loss_history_all': loss_history_all,
        'n_samples': n_samples
    }

    return codes_all, info_all


def get_nonzero(code):
    # return the number of non-zero code
    if isinstance(code, torch.Tensor):
        code = code.detach().cpu().numpy()
    max_abs_code = np.abs(code).max()
    threshold = 0.1 * max_abs_code  # 10% of max
    n_nonzero = np.sum(np.abs(code) > threshold)
    return n_nonzero

def sparse_encode_pytorch_with_shift(
    target: np.ndarray,
    dictionary: np.ndarray,
    max_shift_ms = 1000.0,
    sampling_rate: float = 14.0,
    sparsity_weight: float = 0.001,
    sparsity_type: str = 'L1',
    l1_ratio: float = 0.5,
    n_iterations: int = 10,
    n_steps_shift: int = 100,
    n_steps_code: int = 300,
    lr_shift: float = 0.01,
    lr_code: float = 0.01,
    device: Optional[str] = None,
    verbose: bool = True,
    activation_type: str = 'global',
    activation_function: str = 'exponential',
    use_cyclic_lr: bool = True,
    base_lr_shift: Optional[float] = None,
    max_lr_shift: Optional[float] = None,
    step_size_up_shift: int = 50,
    base_lr_code: Optional[float] = None,
    max_lr_code: Optional[float] = None,
    step_size_up_code: int = 150,
    use_mlflow: bool = False,
    mlflow_run_name: Optional[str] = None,
    init_shift_ms: Optional[np.ndarray] = None,
    shift_print_step = 50,
    code_print_step = 1000,
    early_stop_patience: int = 20,
    early_stop_threshold: float = 1e-7,
    validation_target: Optional[np.ndarray] = None,
    validation_dictionary: Optional[np.ndarray] = None,
    val_early_stop_patience: int = 3,
    val_early_stop_metric: str = 'r2',
    save_best_model: bool = True,
    verbose_validation: bool = True
) -> Tuple[np.ndarray, dict, np.ndarray, nn.Module]:
    """
    PyTorch-based sparse encoding with learnable time shifts and activation.

    This function implements iterative optimization that alternates between:
    1. Optimizing per-neuron time shifts (freeze code + activation)
    2. Optimizing sparse code + activation parameters (freeze shifts)

    The time shifts are learned using Fourier phase shifts, which provide exact
    fractional-sample shifts via frequency domain manipulation.

    Parameters:
    -----------
    target : np.ndarray
        Target signal to decompose, shape (1, n_features) or (n_features,)
    dictionary : np.ndarray
        Dictionary of atoms (unshifted neural patterns), shape (n_components, n_features)
    max_shift_ms : float or list/tuple
        Maximum allowed time shift. Can be:
        - float: shifts range from [0, max_shift_ms] (default: 1000.0)
        - list/tuple [min_shift_ms, max_shift_ms]: arbitrary shift range, including negative values
        Examples: 1000.0 → [0, 1000]ms, [-500, 500] → [-500, 500]ms
    sampling_rate : float
        Sampling frequency in Hz (default: 14.0)
    sparsity_weight : float
        Regularization weight for code sparsity (default: 0.001)
    sparsity_type : str
        Type of sparsity regularization: 'L1', 'L2', or 'elastic_net'
        - 'L1': Promotes true sparsity (many zeros)
        - 'L2': Keeps all coefficients small but non-zero
        - 'elastic_net': Combines L1 and L2 based on l1_ratio
        (default: 'L1')
    l1_ratio : float
        Mixing parameter for elastic net (only used when sparsity_type='elastic_net')
        - 1.0: Pure L1 regularization
        - 0.0: Pure L2 regularization
        - 0.5: Equal mix of L1 and L2 (default: 0.5)
    n_iterations : int
        Number of alternating optimization cycles (default: 10)
    n_steps_shift : int
        Gradient steps per shift optimization phase (default: 100)
    n_steps_code : int
        Gradient steps per code/activation optimization phase (default: 300)
    lr_shift : float
        Learning rate for shift optimization (default: 0.01)
    lr_code : float
        Learning rate for code/activation optimization (default: 0.01)
    device : str, optional
        Device to use ('cuda' or 'cpu'). If None, auto-detect.
    verbose : bool
        Whether to print progress (default: True)
    activation_type : str
        Type of activation: 'global' (single activation after sum),
        'per_neuron' (separate activation per neuron), or 'legacy' (old TwoExpFastEnd)
        Default: 'global'
    activation_function : str
        Type of activation function: 'exponential' or 'sigmoid' (default: 'exponential')
        Only used when activation_type is 'global' or 'per_neuron'
    use_cyclic_lr : bool
        Whether to use cyclic learning rate for both optimizers (default: True)
    base_lr_shift : float, optional
        Minimum learning rate for shift optimizer cyclic LR. If None, uses lr_shift / 10
    max_lr_shift : float, optional
        Maximum learning rate for shift optimizer cyclic LR. If None, uses lr_shift * 10
    step_size_up_shift : int
        Number of steps for half cycle of shift optimizer (default: 50)
    base_lr_code : float, optional
        Minimum learning rate for code optimizer cyclic LR. If None, uses lr_code / 10
    max_lr_code : float, optional
        Maximum learning rate for code optimizer cyclic LR. If None, uses lr_code * 10
    step_size_up_code : int
        Number of steps for half cycle of code optimizer (default: 150)
    use_mlflow : bool
        Whether to log metrics and parameters to MLflow (default: False)
    mlflow_run_name : str, optional
        Name for the MLflow run. If None, MLflow will auto-generate a name
    init_shift_ms : np.ndarray, optional
        Initial shift values in milliseconds, shape (n_components,)
        If None, initializes to middle of allowed range
    early_stop_patience : int
        Number of consecutive steps with loss change below threshold before stopping early
        (default: 20)
    early_stop_threshold : float
        Minimum absolute loss change to consider as improvement. If loss change is below
        this threshold for `early_stop_patience` consecutive steps, optimization stops early
        (default: 1e-7)
    validation_target : np.ndarray, optional
        Validation target signal, shape (1, n_features) or (n_features,).
        If provided along with validation_dictionary, enables validation evaluation
        after each iteration and early stopping based on validation performance.
        Default: None (no validation)
    validation_dictionary : np.ndarray, optional
        Validation dictionary of atoms, shape (n_components, n_features).
        Must have same n_components as training dictionary.
        Default: None (no validation)
    val_early_stop_patience : int
        Number of iterations without validation metric improvement before stopping early.
        Only used if validation data is provided.
        Default: 3
    val_early_stop_metric : str
        Metric to use for validation early stopping: 'r2' (higher is better) or 'mse' (lower is better).
        Only used if validation data is provided.
        Default: 'r2'
    save_best_model : bool
        Whether to checkpoint and restore the best model based on validation metric.
        Only used if validation data is provided.
        Default: True
    verbose_validation : bool
        Whether to print validation metrics during training.
        Only used if validation data is provided.
        Default: True

    Returns:
    --------
    code : np.ndarray
        Sparse code, shape (1, n_components)
    info : dict
        Dictionary containing:
        - 'reconstruction_loss': Final reconstruction loss (MSE)
        - 'total_loss': Final total loss (reconstruction + sparsity)
        - 'n_nonzero': Number of non-zero coefficients (threshold > 0.01)
        - 'variance_explained': Final variance explained (R² = 1 - MSE/var(target))
        - 'loss_history': Dict with 'shift' and 'code' loss lists
        - 'activation_params': Dictionary of learned activation parameters
        - 'shifts_ms': Final learned time shifts in milliseconds
        - 'shifts_ms_history': List of shift values at each iteration
        - 'r2_history': List of variance explained values at each iteration
        - 'train_r2_history': List of train R² values at each iteration
        - 'val_r2_history': List of validation R² values at each iteration (empty if no validation)
        - 'val_mse_history': List of validation MSE values at each iteration (empty if no validation)
        - 'best_val_metric': Best validation metric value (None if no validation)
        - 'best_val_iteration': Iteration index with best validation metric (None if no validation)
        - 'val_early_stopped': Boolean indicating if training was stopped early due to validation
        - 'n_iterations_completed': Number of iterations actually completed
    reconstruction : np.ndarray
        Reconstructed signal, shape (1, n_features)
    model : nn.Module
        Trained model (restored to best checkpoint if validation enabled and save_best_model=True)
    """
    # Handle device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    if verbose:
        print(f"Using device: {device}")
        if isinstance(max_shift_ms, (list, tuple)):
            print(f"Learning time shifts in range [{max_shift_ms[0]}, {max_shift_ms[1]}]ms, sampling_rate={sampling_rate}Hz")
        else:
            print(f"Learning time shifts in range [0, {max_shift_ms}]ms, sampling_rate={sampling_rate}Hz")

    # Start MLflow run if enabled
    if use_mlflow:
        mlflow.set_experiment('sparse_encode_with_shift')
        mlflow.end_run()
        mlflow.start_run(run_name=mlflow_run_name)
        # Log hyperparameters
        mlflow_params = {
            'sampling_rate': sampling_rate,
            'sparsity_weight': sparsity_weight,
            'sparsity_type': sparsity_type,
            'l1_ratio': l1_ratio if sparsity_type == 'elastic_net' else None,
            'n_iterations': n_iterations,
            'n_steps_shift': n_steps_shift,
            'n_steps_code': n_steps_code,
            'lr_shift': lr_shift,
            'lr_code': lr_code,
            'device': device,
            'activation_type': activation_type,
            'activation_function': activation_function,
            'use_cyclic_lr': use_cyclic_lr,
            'n_components': dictionary.shape[0],
            'n_features': dictionary.shape[1]
        }
        # Add shift range parameters
        if isinstance(max_shift_ms, (list, tuple)):
            mlflow_params['min_shift_ms'] = max_shift_ms[0]
            mlflow_params['max_shift_ms'] = max_shift_ms[1]
        else:
            mlflow_params['min_shift_ms'] = 0.0
            mlflow_params['max_shift_ms'] = max_shift_ms

        mlflow.log_params(mlflow_params)
        if use_cyclic_lr:
            mlflow.log_params({
                'base_lr_shift': base_lr_shift if base_lr_shift is not None else lr_shift / 10,
                'max_lr_shift': max_lr_shift if max_lr_shift is not None else lr_shift * 10,
                'step_size_up_shift': step_size_up_shift,
                'base_lr_code': base_lr_code if base_lr_code is not None else lr_code / 10,
                'max_lr_code': max_lr_code if max_lr_code is not None else lr_code * 10,
                'step_size_up_code': step_size_up_code
            })

    # Reshape target if needed
    if target.ndim == 1:
        target = target.reshape(1, -1)

    # Convert to tensors
    target_tensor = torch.FloatTensor(target).to(device)
    dictionary_tensor = torch.FloatTensor(dictionary).to(device)

    # Convert validation data to tensors if provided
    validation_target_tensor = None
    validation_dictionary_tensor = None
    has_validation = False

    if validation_target is not None and validation_dictionary is not None:
        has_validation = True

        # Validate shapes
        if validation_dictionary.shape[0] != dictionary.shape[0]:
            raise ValueError(
                f"Validation dictionary must have same number of components as training. "
                f"Got {validation_dictionary.shape[0]}, expected {dictionary.shape[0]}"
            )

        # Warn if validation set is very small
        val_size = validation_target.size if validation_target.ndim == 1 else validation_target.shape[1]
        train_size = target.size if target.ndim == 1 else target.shape[1]
        val_ratio = val_size / train_size

        if val_ratio < 0.1 and verbose:
            print(f"Warning: Validation set is small ({val_ratio*100:.1f}% of training size)")

        # Reshape if needed
        if validation_target.ndim == 1:
            validation_target = validation_target.reshape(1, -1)

        # Convert to tensors
        validation_target_tensor = torch.FloatTensor(validation_target).to(device)
        validation_dictionary_tensor = torch.FloatTensor(validation_dictionary).to(device)

        if verbose:
            print(f"Validation enabled: {validation_target.shape[1]} features")
            print(f"Early stopping: patience={val_early_stop_patience}, metric={val_early_stop_metric}")

    # Initialize shift dictionary with Fourier phase shifts
    init_shift_tensor = None
    if init_shift_ms is not None:
        init_shift_tensor = torch.FloatTensor(init_shift_ms).to(device)

    shift_dict = FourierShiftDictionary(
        dictionary=dictionary_tensor,
        max_shift_ms=max_shift_ms,
        sampling_rate=sampling_rate,
        init_shift_ms=init_shift_tensor
    ).to(device)

    # Initialize sparse coding model with shifts
    model = SparseCodingWithShifts(
        shift_dictionary=shift_dict,
        n_neurons=dictionary.shape[0],
        activation_type=activation_type,
        activation_function=activation_function
    ).to(device)

    if verbose:
        # Count total parameters in model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total trainable parameters in model: {total_params}")

    # Training history
    history = {
        'loss_shift': [],
        'loss_code': [],
        'shifts_ms_history': [],
        'r2_history': [],
        'train_r2_history': [],
        'val_r2_history': [],
        'val_mse_history': []
    }

    # Early stopping tracking for validation
    best_val_metric = -np.inf if val_early_stop_metric == 'r2' else np.inf
    best_val_iteration = -1
    val_no_improvement_count = 0
    best_model_state = None

    # ----------------------------------------------------------------
    # Setup optimizers and schedulers (persistent across iterations)
    # ----------------------------------------------------------------

    # Get parameters to optimize based on activation type
    if activation_type == 'global' or activation_type == 'legacy':
        params_to_optimize_code = [
            model.code,
            *model.act.parameters()
        ]
    elif activation_type == 'per_neuron':
        params_to_optimize_code = [
            model.code,
            *[p for act in model.activations for p in act.parameters()]
        ]
    else:
        params_to_optimize_code = [model.code]

    # Create optimizers
    optimizer_code = torch.optim.Adam(params_to_optimize_code, lr=lr_code)
    optimizer_shift = torch.optim.Adam(model.shift_dictionary.parameters(), lr=lr_shift)

    # Create learning rate schedulers
    scheduler_code = None
    scheduler_shift = None

    if use_cyclic_lr:
        # Set default base_lr and max_lr if not provided
        _base_lr_code = base_lr_code if base_lr_code is not None else lr_code / 10
        _max_lr_code = max_lr_code if max_lr_code is not None else lr_code * 10
        _base_lr_shift = base_lr_shift if base_lr_shift is not None else lr_shift / 10
        _max_lr_shift = max_lr_shift if max_lr_shift is not None else lr_shift * 10

        scheduler_code = optim.lr_scheduler.CyclicLR(
            optimizer_code,
            base_lr=_base_lr_code,
            max_lr=_max_lr_code,
            step_size_up=step_size_up_code,
            mode='triangular2',  # triangular2 decreases max_lr after each cycle
            cycle_momentum=False
        )

        scheduler_shift = optim.lr_scheduler.CyclicLR(
            optimizer_shift,
            base_lr=_base_lr_shift,
            max_lr=_max_lr_shift,
            step_size_up=step_size_up_shift,
            mode='triangular2',  # triangular2 decreases max_lr after each cycle
            cycle_momentum=False
        )

        if verbose:
            print(f"Using CyclicLR for code optimizer: base_lr={_base_lr_code:.2e}, max_lr={_max_lr_code:.2e}, step_size_up={step_size_up_code}")
            print(f"Using CyclicLR for shift optimizer: base_lr={_base_lr_shift:.2e}, max_lr={_max_lr_shift:.2e}, step_size_up={step_size_up_shift}")

    # Helper function for validation evaluation
    def _evaluate_on_validation(
        model_to_eval: nn.Module,
        val_dictionary: torch.Tensor,
        val_target: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Evaluate model on validation data.

        Returns:
            tuple: (validation_r2, validation_mse)
        """
        with torch.no_grad():
            model_to_eval.eval()

            # Create validation shift dictionary with same learned shifts
            val_shift_dict = FourierShiftDictionary(
                dictionary=val_dictionary,
                max_shift_ms=max_shift_ms,
                sampling_rate=sampling_rate
            ).to(device)

            # Copy learned shift parameters
            val_shift_dict.shift_logits.data = model_to_eval.shift_dictionary.shift_logits.data.clone()

            # Create validation model
            val_model = SparseCodingWithShifts(
                shift_dictionary=val_shift_dict,
                n_neurons=val_dictionary.shape[0],
                activation_type=activation_type
            ).to(device)

            # Copy learned code and activation parameters
            val_model.code.data = model_to_eval.code.data.clone()

            if activation_type == 'global' or activation_type == 'legacy':
                val_model.act.load_state_dict(model_to_eval.act.state_dict())
            elif activation_type == 'per_neuron':
                for i, act in enumerate(model_to_eval.activations):
                    val_model.activations[i].load_state_dict(act.state_dict())

            # Forward pass
            val_reconstruction = val_model()

            # Compute metrics
            val_mse = F.mse_loss(val_reconstruction, val_target).item()
            val_var = val_target.var(unbiased=False).item()
            val_r2 = 1 - (val_mse / val_var) if val_var > 0 else 0.0

            model_to_eval.train()  # Restore training mode

            return val_r2, val_mse

    # Iterative optimization
    for iter_idx in range(n_iterations):

        if verbose:
            print(f"\n=== Iteration {iter_idx + 1}/{n_iterations} ===")


        # ----------------------------------------------------------------
        # Phase A: Optimize CODE + ACTIVATION (freeze shifts)
        # ----------------------------------------------------------------

        # Early stopping tracking for code optimization
        code_loss_history = []
        code_no_improvement_count = 0

        for step in range(n_steps_code):
            optimizer_code.zero_grad()

            reconstruction = model()
            mse_loss = F.mse_loss(reconstruction, target_tensor)

            # Compute sparsity loss based on type
            if sparsity_type == 'L2':
                sparsity_loss = sparsity_weight * (model.code ** 2).sum()
            elif sparsity_type == 'elastic_net':
                l1_loss = torch.abs(model.code).sum()
                l2_loss = (model.code ** 2).sum()
                sparsity_loss = sparsity_weight * (l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss)
            else:  # L1 (default)
                sparsity_loss = sparsity_weight * torch.abs(model.code).sum()

            loss = mse_loss + sparsity_loss
            loss.backward()
            optimizer_code.step()

            # Update learning rate if using scheduler
            if scheduler_code is not None:
                scheduler_code.step()

            # Early stopping check
            current_loss = loss.item()
            code_loss_history.append(current_loss)

            if len(code_loss_history) > 1:
                loss_change = abs(code_loss_history[-2] - current_loss)
                if loss_change < early_stop_threshold:
                    code_no_improvement_count += 1
                else:
                    code_no_improvement_count = 0

                if code_no_improvement_count >= early_stop_patience:
                    if verbose:
                        print(f"  Code optimization converged at step {step+1}/{n_steps_code} (loss change < {early_stop_threshold} for {early_stop_patience} steps)")
                    break

            if step % code_print_step == 0 and verbose:
                # Compute variance explained (R^2 = 1 - MSE / var(target))
                var_target = target_tensor.var(unbiased=False)
                if var_target.item() != 0:
                    variance_explained = 1 - (mse_loss.detach() / var_target)
                else:
                    variance_explained = torch.tensor(0.0, device=mse_loss.device)
                current_lr = optimizer_code.param_groups[0]['lr']
                n_nonzero = get_nonzero(model.code)

                print(f"  Code opt step {step}/{n_steps_code}, loss: {loss.item():.6f}, variance explained: {variance_explained.item():.4f}, LR: {current_lr:.2e}")

                # Log metrics to MLflow at each print step
                if use_mlflow:
                    global_step = iter_idx * n_steps_code + step
                    mlflow.log_metrics({
                        'code_loss': loss.item(),
                        'reconstruction_loss': mse_loss.item(),
                        'sparsity_loss': sparsity_loss.item(),
                        'variance_explained': variance_explained.item(),
                        'n_nonzero': int(n_nonzero),
                        'sparsity_ratio': float(n_nonzero) / dictionary.shape[0],
                        'code_lr': current_lr
                    }, step=global_step)

        history['loss_code'].append(loss.item())

        # ----------------------------------------------------------------
        # Phase B: Optimize SHIFTS (freeze code + activation)
        # ----------------------------------------------------------------

        # Early stopping tracking for shift optimization
        shift_loss_history = []
        shift_no_improvement_count = 0

        for step in range(n_steps_shift):
            optimizer_shift.zero_grad()

            reconstruction = model()
            mse_loss = F.mse_loss(reconstruction, target_tensor)

            loss = mse_loss
            loss.backward()
            optimizer_shift.step()

            # Update learning rate if using scheduler
            if scheduler_shift is not None:
                scheduler_shift.step()

            # Early stopping check
            current_loss = loss.item()
            shift_loss_history.append(current_loss)

            if len(shift_loss_history) > 1:
                loss_change = abs(shift_loss_history[-2] - current_loss)
                if loss_change < early_stop_threshold:
                    shift_no_improvement_count += 1
                else:
                    shift_no_improvement_count = 0

                if shift_no_improvement_count >= early_stop_patience:
                    if verbose:
                        print(f"  Shift optimization converged at step {step+1}/{n_steps_shift} (loss change < {early_stop_threshold} for {early_stop_patience} steps)")
                    break

            if step % shift_print_step == 0 and verbose:
                current_lr = optimizer_shift.param_groups[0]['lr']

                # Compute shift statistics
                with torch.no_grad():
                    shifts_ms = model.shift_dictionary.get_shifts_ms().cpu().numpy()
                    reconstruction = model()
                    var_target = target_tensor.var(unbiased=False)
                    variance_explained = 1 - (mse_loss.detach() / var_target)

                print(f"  Shift opt step {step}/{n_steps_shift}, loss: {loss.item():.6f}, LR: {current_lr:.2e}")

                # Log metrics to MLflow at each print step
                if use_mlflow:
                    global_step = iter_idx * n_steps_shift + step
                    mlflow.log_metrics({
                        'shift_loss': loss.item(),
                        'shift_lr': current_lr,
                        'variance_explained': variance_explained.item(),
                        'mean_shift_ms': float(shifts_ms.mean()),
                        'std_shift_ms': float(shifts_ms.std()),
                        'min_shift_ms': float(shifts_ms.min()),
                        'max_shift_ms': float(shifts_ms.max())
                    }, step=global_step)

        history['loss_shift'].append(loss.item())

        # Record shifts and variance explained (R²) for history tracking
        with torch.no_grad():
            reconstruction = model()
            variance_explained = 1 - F.mse_loss(reconstruction, target_tensor) / target_tensor.var(unbiased=False)
            shifts_ms = model.shift_dictionary.get_shifts_ms().cpu().numpy().copy()

            history['shifts_ms_history'].append(shifts_ms)
            history['r2_history'].append(variance_explained.item())

            # Add train R² tracking
            train_r2 = variance_explained.item()
            history['train_r2_history'].append(train_r2)

            if verbose:
                n_nonzero = get_nonzero(model.code)
                print(f"  Train R² = {train_r2:.4f}")
                print(f"  Mean shift: {shifts_ms.mean():.1f} ms (std: {shifts_ms.std():.1f} ms)")
                print(f"  Non-zero coefficients: {n_nonzero}/{dictionary.shape[0]}")

            # Validation evaluation
            if has_validation:
                val_r2, val_mse = _evaluate_on_validation(
                    model_to_eval=model,
                    val_dictionary=validation_dictionary_tensor,
                    val_target=validation_target_tensor
                )

                history['val_r2_history'].append(val_r2)
                history['val_mse_history'].append(val_mse)

                if verbose_validation:
                    print(f"  Val R² = {val_r2:.4f}, Val MSE = {val_mse:.6f}")
                    print(f"  Train-Val gap = {train_r2 - val_r2:.4f}")

                # Log validation metrics to MLflow
                if use_mlflow:
                    mlflow.log_metrics({
                        'train_r2': train_r2,
                        'val_r2': val_r2,
                        'val_mse': val_mse,
                        'train_val_gap': train_r2 - val_r2
                    }, step=iter_idx)

                # Early stopping and checkpointing logic
                current_val_metric = val_r2 if val_early_stop_metric == 'r2' else val_mse
                is_improvement = False

                if val_early_stop_metric == 'r2':
                    # Higher is better
                    if current_val_metric > best_val_metric + 1e-6:  # Small epsilon for numerical stability
                        is_improvement = True
                else:  # 'mse'
                    # Lower is better
                    if current_val_metric < best_val_metric - 1e-6:
                        is_improvement = True

                if is_improvement:
                    best_val_metric = current_val_metric
                    best_val_iteration = iter_idx
                    val_no_improvement_count = 0

                    # Save best model state
                    if save_best_model:
                        import copy
                        best_model_state = copy.deepcopy(model.state_dict())

                    if verbose_validation:
                        metric_name = 'R²' if val_early_stop_metric == 'r2' else 'MSE'
                        print(f"  ✓ New best validation {metric_name}: {current_val_metric:.4f}")
                else:
                    val_no_improvement_count += 1

                    if verbose_validation and val_no_improvement_count > 0:
                        print(f"  No improvement for {val_no_improvement_count}/{val_early_stop_patience} iterations")

                # Check early stopping
                if val_no_improvement_count >= val_early_stop_patience:
                    if verbose:
                        metric_name = 'R²' if val_early_stop_metric == 'r2' else 'MSE'
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered!")
                        print(f"No improvement in validation {metric_name} for {val_early_stop_patience} iterations")
                        print(f"Best validation {metric_name}: {best_val_metric:.4f} at iteration {best_val_iteration+1}")
                        print(f"{'='*60}\n")
                    break  # Exit iteration loop

    # Restore best model if validation was used and checkpointing enabled
    if has_validation and save_best_model and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            metric_name = 'R²' if val_early_stop_metric == 'r2' else 'MSE'
            print(f"\nRestored model from iteration {best_val_iteration+1} (best validation {metric_name}: {best_val_metric:.4f})")

    # Get final results
    with torch.no_grad():
        code_final = model.get_code().cpu().numpy()
        reconstruction_final = model().cpu().numpy()
        reconstruction_loss_final = F.mse_loss(model(), target_tensor).item()
        shifts_ms_final = model.shift_dictionary.get_shifts_ms().cpu().numpy()

    # Count non-zero coefficients
    n_nonzero = get_nonzero(code_final)

    # Calculate final variance explained
    final_variance_explained = 1 - (reconstruction_loss_final / np.var(target))

    # Compile info dictionary
    info = {
        'reconstruction_loss': reconstruction_loss_final,
        'total_loss': history['loss_code'][-1],
        'n_nonzero': n_nonzero,
        'variance_explained': final_variance_explained,
        'loss_history': {
            'shift': history['loss_shift'],
            'code': history['loss_code']
        },
        'activation_params': model.get_activation_params(),
        'shifts_ms': shifts_ms_final,
        'shifts_ms_history': history['shifts_ms_history'],
        'r2_history': history['r2_history'],
        # Add validation info
        'train_r2_history': history['train_r2_history'],
        'val_r2_history': history['val_r2_history'] if has_validation else [],
        'val_mse_history': history['val_mse_history'] if has_validation else [],
        'best_val_metric': best_val_metric if has_validation else None,
        'best_val_iteration': best_val_iteration if has_validation else None,
        'val_early_stopped': (has_validation and val_no_improvement_count >= val_early_stop_patience),
        'n_iterations_completed': len(history['r2_history'])
    }

    # Final MLflow logging
    if use_mlflow:
        try:
            # Log final metrics
            final_variance_explained = 1 - (reconstruction_loss_final / np.var(target))
            mlflow_metrics = {
                'final_reconstruction_loss': reconstruction_loss_final,
                'final_total_loss': history['loss_code'][-1],
                'final_n_nonzero': int(n_nonzero),
                'final_sparsity_ratio': float(n_nonzero) / dictionary.shape[0],
                'final_variance_explained': float(final_variance_explained),
                'final_mean_shift_ms': float(shifts_ms_final.mean()),
                'final_std_shift_ms': float(shifts_ms_final.std())
            }

            # Add validation metrics
            if has_validation:
                mlflow_metrics.update({
                    'final_train_r2': history['train_r2_history'][-1],
                    'final_val_r2': history['val_r2_history'][-1] if history['val_r2_history'] else None,
                    'best_val_metric': best_val_metric,
                    'best_val_iteration': best_val_iteration + 1,
                    'val_early_stopped': info['val_early_stopped'],
                    'n_iterations_completed': info['n_iterations_completed']
                })

            mlflow.log_metrics(mlflow_metrics)

            # Log the model
            mlflow.pytorch.log_model(model, "model")

        finally:
            mlflow.end_run()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimization complete!")
        print(f"Final Variance Explained: {info['r2_history'][-1]:.4f}")
        print(f"Final reconstruction loss: {reconstruction_loss_final:.6f}")
        print(f"Non-zero coefficients: {n_nonzero}/{dictionary.shape[0]}")
        print(f"Shift range: [{shifts_ms_final.min():.1f}, {shifts_ms_final.max():.1f}] ms")
        print(f"Mean shift: {shifts_ms_final.mean():.1f} ± {shifts_ms_final.std():.1f} ms")
        print(f"{'='*60}")

    return code_final, info, reconstruction_final, model


def sparse_encode_pytorch_with_shift_trials(
    target: np.ndarray,
    dictionary: np.ndarray,
    fixed_shifts: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, dict, np.ndarray, nn.Module]:
    """
    Wrapper for sparse_encode_pytorch_with_shift that accepts trial-stacked format.

    This function provides a cleaner API for cross-validation by accepting
    trial-stacked inputs instead of concatenated trials. Internally, it flattens
    the trials, calls the existing sparse_encode_pytorch_with_shift function,
    and reshapes the output back to trial-stacked format.

    Parameters
    ----------
    target : np.ndarray
        Target signal, shape (time, n_trials)
    dictionary : np.ndarray
        Dictionary atoms, shape (n_neurons, time, n_trials)
    fixed_shifts : np.ndarray, optional
        Pre-learned shifts for test mode, shape (n_neurons,) in milliseconds.
        If provided, will be passed as init_shift_ms to initialize shifts.
    **kwargs
        All other parameters are passed through to sparse_encode_pytorch_with_shift.
        Common parameters include:
        - max_shift_ms: Maximum shift range (default: 1000.0 or [-200, 200])
        - sampling_rate: Sampling frequency in Hz (default: 14.0)
        - sparsity_weight: Sparsity regularization weight (default: 0.001)
        - sparsity_type: Type of sparsity ('L1', 'L2', 'elastic_net')
        - n_iterations: Number of alternating optimization iterations (default: 10)
        - n_steps_code: Steps for code optimization (default: 300)
        - n_steps_shift: Steps for shift optimization (default: 100)
        - device: 'cpu' or 'cuda' (default: auto-detect)
        - verbose: Print progress (default: True)
        - validation_target: Validation target, shape (time, n_trials_val) (default: None)
        - validation_dictionary: Validation dictionary, shape (n_neurons, time, n_trials_val) (default: None)
        - val_early_stop_patience: Early stopping patience for validation (default: 3)
        - val_early_stop_metric: Metric for early stopping: 'r2' or 'mse' (default: 'r2')
        - save_best_model: Whether to restore best model (default: True)
        - verbose_validation: Print validation metrics (default: True)

    Returns
    -------
    code : np.ndarray
        Sparse codes, shape (1, n_neurons) - shared across all trials
    info : dict
        Information dictionary containing:
        - 'variance_explained': R² score across all trials
        - 'reconstruction_loss': MSE across all trials
        - 'shifts_ms': Learned time shifts, shape (n_neurons,)
        - 'n_nonzero': Number of active neurons
        - 'n_trials': Number of trials (added by wrapper)
        - 'n_timepoints': Timepoints per trial (added by wrapper)
        - ... other metrics from original function
    reconstruction : np.ndarray
        Reconstructed signal, shape (time, n_trials) - reshaped to match input
    model : nn.Module
        Trained model instance (SparseCodingWithShifts)

    Examples
    --------
    # Training with trial-stacked data
    >>> code, info, recon, model = sparse_encode_pytorch_with_shift_trials(
    ...     target=target_train,          # (time, n_train_trials)
    ...     dictionary=dict_train,        # (n_neurons, time, n_train_trials)
    ...     max_shift_ms=[-200, 200],
    ...     sparsity_weight=5e-3
    ... )

    # Testing with fixed shifts from training
    >>> code_test, info_test, recon_test, model_test = sparse_encode_pytorch_with_shift_trials(
    ...     target=target_test,
    ...     dictionary=dict_test,
    ...     fixed_shifts=info['shifts_ms'],  # Use learned shifts
    ...     max_shift_ms=[-200, 200],
    ...     sparsity_weight=5e-3
    ... )

    Notes
    -----
    - The model architecture (shared code and shifts) is identical to the original function
    - Expected performance is identical to concatenated approach
    - This wrapper only changes the input/output format for convenience
    """
    # Extract dimensions
    n_neurons, n_timepoints, n_trials = dictionary.shape

    # Validate target shape
    if target.shape != (n_timepoints, n_trials):
        raise ValueError(
            f"Target shape {target.shape} doesn't match expected "
            f"(time={n_timepoints}, n_trials={n_trials})"
        )

    # ===== FLATTEN: (n_neurons, time, trials) → (n_neurons, trials*time) =====
    # IMPORTANT: Must match original concatenation order: (cluID, trial, time) → (cluID, trial*time)
    # Original does: atoms_smooth (time, cluID, trial) → transpose([1,2,0]) → (cluID, trial, time) → reshape
    # We have: (n_neurons, time, trial) → transpose to (n_neurons, trial, time) → reshape
    dict_flat = dictionary.transpose([0, 2, 1])  # (n_neurons, trials, time)
    dict_flat = dict_flat.reshape(n_neurons, n_trials * n_timepoints)  # (n_neurons, trials*time)

    # Target: (time, trials) → (trials, time) → (1, trials*time)
    target_flat = target.T.reshape(1, n_trials * n_timepoints)  # (1, trials*time)

    # ===== HANDLE VALIDATION DATA IF PROVIDED =====
    # Check if validation data was passed and flatten it as well
    if 'validation_target' in kwargs and kwargs['validation_target'] is not None:
        val_target = kwargs['validation_target']
        val_dictionary = kwargs['validation_dictionary']

        # Extract validation dimensions
        n_neurons_val, n_timepoints_val, n_trials_val = val_dictionary.shape

        # Flatten validation data (same process as training)
        val_dict_flat = val_dictionary.transpose([0, 2, 1])  # (n_neurons, trials_val, time)
        val_dict_flat = val_dict_flat.reshape(n_neurons_val, n_trials_val * n_timepoints_val)

        val_target_flat = val_target.T.reshape(1, n_trials_val * n_timepoints_val)

        # Replace in kwargs with flattened versions
        kwargs['validation_target'] = val_target_flat
        kwargs['validation_dictionary'] = val_dict_flat

    # ===== CALL EXISTING FUNCTION =====
    # If fixed_shifts provided, pass as init_shift_ms to initialize shifts
    if fixed_shifts is not None:
        kwargs['init_shift_ms'] = fixed_shifts

    code, info, recon_flat, model = sparse_encode_pytorch_with_shift(
        target=target_flat,             # (1, time*trials)
        dictionary=dict_flat,           # (n_neurons, time*trials)
        **kwargs
    )

    # ===== RESHAPE OUTPUT: (1, trials*time) → (time, trials) =====
    # Original reconstruction is (1, trials*time) following (trial, time) order
    # Need to reshape to (trials, time) then transpose to (time, trials)
    recon_reshaped = recon_flat.reshape(n_trials, n_timepoints).T  # (time, trials)

    # ===== UPDATE INFO DICT =====
    info['n_trials'] = n_trials
    info['n_timepoints'] = n_timepoints

    return code, info, recon_reshaped, model


def sparse_encode_sklearn_baseline(
    target: np.ndarray,
    dictionary: np.ndarray,
    code = None,
    sparsity_weight: float = 0.001,
    algorithm: str = 'lasso_lars',
    verbose: bool = True,
) -> Tuple[np.ndarray, dict, np.ndarray, None]:
    """
    Simplified baseline sparse encoding using sklearn's sparse_encode.

    This function provides a simple baseline for comparison with the PyTorch version,
    using standard sklearn sparse coding without time shifts or activation learning.

    Parameters:
    -----------
    target : np.ndarray
        Target signal to decompose, shape (1, n_features) or (n_features,)
    dictionary : np.ndarray
        Dictionary of atoms (neural patterns), shape (n_components, n_features)
    sparsity_weight : float
        Regularization weight for sparsity (alpha parameter in sklearn)
        Default: 0.001
    algorithm : str
        Algorithm to use for sparse coding:
        - 'lasso_lars': LARS algorithm (fast, default)
        - 'lasso_cd': Coordinate descent (good for large datasets)
        - 'lars': Least angle regression (no L1 penalty)
        - 'threshold': Simple thresholding
        - 'omp': Orthogonal matching pursuit
        Default: 'lasso_lars'
    verbose : bool
        Whether to print progress information
        Default: True

    Returns:
    --------
    code : np.ndarray
        Sparse code, shape (1, n_components)
    info : dict
        Dictionary containing:
        - 'reconstruction_loss': MSE reconstruction loss
        - 'total_loss': Total loss (reconstruction + sparsity penalty)
        - 'n_nonzero': Number of non-zero coefficients (threshold > 0.01)
        - 'variance_explained': R² = 1 - MSE/var(target)
    reconstruction : np.ndarray
        Reconstructed signal, shape (1, n_features)
    activation_module : None
        Returns None (no activation module in baseline)

    Notes:
    ------
    - This is a simple baseline without time shifts or learnable activations
    - Uses the same dictionary and target format as sparse_encode_pytorch_with_shift
    - Returns the same output structure for easy comparison
    - Much faster than PyTorch version but less flexible
    """

    if verbose:
        print(f"{'='*60}")
        print(f"Sklearn Baseline Sparse Encoding")
        print(f"{'='*60}")
        print(f"Target shape: {target.shape}")
        print(f"Dictionary shape: {dictionary.shape}")
        print(f"Sparsity weight (alpha): {sparsity_weight}")
        print(f"Algorithm: {algorithm}")

    # Reshape target if needed
    if target.ndim == 1:
        target = target.reshape(1, -1)

    # sklearn sparse_encode expects:
    # - X: data matrix (n_samples, n_features)
    # - dictionary: (n_components, n_features) - already in correct format
    # - alpha: regularization parameter (our sparsity_weight)

    # Perform sparse encoding
    # only do the training again is code is not provided
    # otherwise just do the reconstruction
    if code is None:
        code = sparse_encode(
            X=target,
            dictionary=dictionary,
            algorithm=algorithm,
            alpha=sparsity_weight,
            n_jobs=1,
            check_input=True,
            verbose=0 if not verbose else 1
        )

    # Compute reconstruction
    reconstruction = code @ dictionary

    # Compute metrics
    mse = np.mean((target - reconstruction) ** 2)
    target_var = np.var(target)
    variance_explained = 1.0 - (mse / target_var) if target_var > 0 else 0.0

    # Count non-zero coefficients (using same threshold as PyTorch version)
    n_nonzero = np.sum(np.abs(code) > 0.01)

    # Compute total loss (reconstruction + L1 penalty)
    l1_penalty = sparsity_weight * np.sum(np.abs(code))
    total_loss = mse + l1_penalty

    # Prepare info dictionary
    info = {
        'reconstruction_loss': float(mse),
        'total_loss': float(total_loss),
        'n_nonzero': int(n_nonzero),
        'variance_explained': float(variance_explained),
        'sparsity': float(n_nonzero / code.shape[1]),  # fraction of non-zero
        'algorithm': algorithm,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Reconstruction loss (MSE): {mse:.6f}")
        print(f"  Variance explained (R²): {variance_explained:.4f}")
        print(f"  Non-zero coefficients: {n_nonzero}/{code.shape[1]} ({100*n_nonzero/code.shape[1]:.1f}%)")
        print(f"  L1 penalty: {l1_penalty:.6f}")
        print(f"  Total loss: {total_loss:.6f}")
        print(f"  Code range: [{code.min():.4f}, {code.max():.4f}]")
        print(f"  Code mean: {code.mean():.4f} ± {code.std():.4f}")
        print(f"{'='*60}")

    return code, info, reconstruction, None


# Example usage and test function
def test_sparse_encode():
    """Test function to demonstrate usage."""
    print("Testing PyTorch sparse encoding with sigmoid transformation...")

    # Create synthetic data
    n_features = 100
    n_components = 20

    # Random dictionary of sine waves with different frequencies
    time = np.linspace(0, 1, n_features)
    dictionary = np.zeros((n_components, n_features))
    for i in range(n_components):
        freq = (i + 1) * 2  # Different frequencies for each component
        dictionary[i] = np.sin(2 * np.pi * freq * time)
    
    # Normalize dictionary atoms
    dictionary = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)
    
    plt.plot(dictionary)

    # Sparse true code
    true_code = np.zeros((1, n_components))
    active_components = np.random.choice(n_components, size=5, replace=False)
    true_code[0, active_components] = np.random.rand(5)

    # Target signal: sum of selected sine waves with noise
    # target = true_code @ dictionary + 0.05 * np.random.randn(1, n_features)
    
    target = torch.sigmoid(torch.tensor(true_code @ dictionary)).numpy()

    # Encode
    code, info, reconstruction = sparse_encode_pytorch(
        target,
        dictionary,
        sparsity_weight=1e-6,
        learning_rate=1e-3,
        n_iterations=6000,
        verbose=True
    )

    print(f"\nTrue non-zero components: {len(active_components)}")
    print(f"Recovered non-zero components: {info['n_nonzero']}")
    print(f"Final reconstruction loss: {info['reconstruction_loss']:.6f}")

    _, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Loss history
    axes[0].plot(info['loss_history'])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    # Code comparison
    axes[1].stem(np.arange(n_components), true_code[0], linefmt='b-',
                 markerfmt='bo', basefmt=' ', label='True')
    axes[1].stem(np.arange(n_components), code[0], linefmt='r-',
                 markerfmt='ro', basefmt=' ', label='Recovered')
    axes[1].set_xlabel('Component Index')
    axes[1].set_ylabel('Code Value')
    axes[1].set_title('True vs Recovered Code')
    axes[1].legend()
    axes[1].grid(True)

    # Original vs Reconstructed curve
    axes[2].plot(time, target[0], 'b-', label='Original', linewidth=2)
    axes[2].plot(time, reconstruction[0], 'r--', label='Reconstructed', linewidth=2)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Signal Value')
    axes[2].set_title('Original vs Reconstructed Signal')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    
    return dictionary, code, reconstruction, target, true_code




def augment_atoms_with_shifts(atoms_stack, xr_session, max_shift_ms=700, shift_step=2,
                               mode='append', shift_amount=None):
    """
    Augment neural atoms with time-shifted versions.

    Parameters:
    -----------
    atoms_stack : np.ndarray
        Original atoms array of shape (n_neurons, n_timepoints)
    xr_session : xr.Dataset
        Session dataset containing event_time coordinate
    max_shift_ms : int
        Maximum shift in milliseconds (default: 700)
    shift_step : int
        Step size for shifts in samples (default: 2)
    mode : str
        'append': Stack all time-shifted versions together (default)
        'replace': Return only one shifted version with same shape as input
    shift_amount : int or None
        For 'replace' mode: specific shift in samples to apply.
        If None, randomly selects a shift from 0 to max_shift_samples.
        Ignored in 'append' mode.

    Returns:
    --------
    atoms_augmented : np.ndarray
        In 'append' mode: augmented atoms array with all time shifts stacked
        In 'replace' mode: single shifted version with same shape as input
    """
    # Get sampling rate from xr_session.event_time
    event_time_diff = np.diff(xr_session.event_time.values)
    sampling_rate = 1000 / event_time_diff[0]  # Convert ms to Hz
    # print(f'Sampling rate: {sampling_rate:.2f} Hz')

    # Calculate maximum shift in samples
    max_shift_samples = int(max_shift_ms * sampling_rate / 1000)

    if mode == 'replace':
        # Replace mode: return a single shifted version
        if shift_amount is None:
            raise ValueError("shift_amount must be specified in 'replace' mode")
        else:
            shift = min(shift_amount, max_shift_samples)  # Ensure shift doesn't exceed max

        if shift == 0:
            atoms_augmented = atoms_stack.copy()
            print(f'Replace mode: No shift applied (shift=0)')
        else:
            # Apply the single shift
            atoms_augmented = np.zeros_like(atoms_stack)
            atoms_augmented[:, shift:] = atoms_stack[:, :-shift]
            print(f'Replace mode: Applied shift of {shift} samples ({shift * 1000 / sampling_rate:.1f} ms)')

        return atoms_augmented

    elif mode == 'append':
        # Append mode: stack all shifted versions (original behavior)
        # Create list to store augmented atoms
        augmented_atoms = [atoms_stack]

        # Add time-shifted versions
        for shift in range(1, max_shift_samples + 1, shift_step):
            # Shift each atom to the right by padding with zeros on the left and truncating on the right
            shifted_atoms = np.zeros_like(atoms_stack)
            shifted_atoms[:, shift:] = atoms_stack[:, :-shift]
            augmented_atoms.append(shifted_atoms)

        # Stack all augmented atoms
        atoms_augmented = np.vstack(augmented_atoms)

        print(f'Append mode: Original number of atoms: {atoms_stack.shape[0]}')
        print(f'Append mode: Augmented number of atoms: {atoms_augmented.shape[0]}')

        return atoms_augmented

    else:
        raise ValueError(f"mode must be 'append' or 'replace', got '{mode}'")
    
    
def plot_top_neurons(codes, atoms, target, trial_outcome, event_time, shift=None, outcome_filter='success', n_neurons=5):
    """
    Plot the top neurons based on code weights for a specific trial outcome.

    Parameters:
    -----------
    codes : array-like
        Code weights for each neuron (excluding intercept)
    atoms : ndarray
        Neural firing rate data (time x neurons x trials)
    target : ndarray
        Target signal (time x trials)
    trial_outcome : array-like
        Trial outcome labels
    event_time : array-like
        Time points for plotting
    shift : int or None
        Time shift in samples to apply to atoms (shifts right by padding left with zeros)
    outcome_filter : str
        Trial outcome to filter ('success', 'aborted', etc.)
    n_neurons : int
        Number of top neurons to plot
    """
    sort_idx = np.argsort(-np.abs(codes.ravel()))

    # Apply shift to atoms if provided
    if shift is None:
        shift = 0
    
    if not isinstance(shift, np.ndarray):
        shift = np.ones_like(codes)*shift
    else:
        # make sure the length of shift and code are the same
        assert len(shift) == len(codes), "Length of shift array must match number of neurons in codes"        
        

    # Create subplots
    n_cols = min(3, n_neurons)
    n_rows = int(np.ceil(n_neurons / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*3))

    # Ensure axes is always a flat array for consistent indexing
    if n_neurons == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i in range(n_neurons):
        n_idx = sort_idx[i]
        outcome_mask = (trial_outcome == outcome_filter)
        # atom2plot = normalize(atoms[:, n_idx, outcome_mask], axis=0)
        atom2plot = atoms[:, n_idx, outcome_mask]

        # Apply time shift to the waveform
        neuron_shift = int(shift[n_idx])
        waveform = atom2plot.mean(axis=1)

        if neuron_shift > 0:
            # Positive shift: neuron fires later, shift waveform to the right
            # Truncate beginning and end to keep valid data only
            shifted_waveform = waveform[:-neuron_shift]
            shifted_event_time = event_time[neuron_shift:]
        elif neuron_shift < 0:
            # Negative shift: neuron fires earlier, shift waveform to the left
            # Truncate beginning and end to keep valid data only
            shifted_waveform = waveform[-neuron_shift:]
            shifted_event_time = event_time[:neuron_shift]
        else:
            # No shift
            shifted_waveform = waveform
            shifted_event_time = event_time

        axes.flat[i].plot(shifted_event_time, shifted_waveform)

        axes.flat[i].axvline(0, ls='--', color='r')
        axes.flat[i].set_title(f'Neuron {i+1}({codes[n_idx]:.2f})')
        axes.flat[i].set_xlabel('Time')
        if i == 0:
            axes.flat[i].set_ylabel('Normalized FR')

        # Create twin axis for DA signal
        ax2 = axes.flat[i].twinx()
        signal = target[:, outcome_mask].mean(axis=1)
        ax2.plot(event_time, signal, color='r', alpha=0.7)
        if i == n_neurons - 1:  # Only label the last subplot
            ax2.set_ylabel('Normalized DA', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add shift information to subplot title
        time_shift_ms = np.diff(event_time)[0] * neuron_shift
        axes.flat[i].set_title(f'Neuron {i+1}({codes[n_idx]:.3f}), shift: {time_shift_ms:.1f} ms')

    plt.tight_layout()
    plt.show()
    return fig


class WindowedTimeSeriesDataset(Dataset):
    """
    Dataset for windowing time series data into non-overlapping segments.

    Used for batched training where each window is processed independently
    but all windows share the same sparse code and shift parameters.
    """

    def __init__(
        self,
        target: torch.Tensor,
        window_size: int,
        drop_last: bool = True
    ):
        """
        Args:
            target: Full target signal, shape (1, n_timepoints) or (n_timepoints,)
            window_size: Number of timepoints per window
            drop_last: If True, drop the last window if it's incomplete
        """
        super().__init__()

        # Ensure target is 2D
        if target.ndim == 1:
            target = target.unsqueeze(0)

        self.target = target  # Shape: (1, n_timepoints)
        self.window_size = window_size
        self.n_timepoints = target.shape[1]

        # Calculate number of windows
        self.n_windows = self.n_timepoints // window_size

        if not drop_last and self.n_timepoints % window_size != 0:
            self.n_windows += 1

        # Precompute window indices for efficiency
        self.window_starts = [i * window_size for i in range(self.n_windows)]

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a window of the target signal.

        Args:
            idx: Window index

        Returns:
            target_window: Shape (1, window_size) or smaller for last window
            start_idx: Starting index in full signal
            end_idx: Ending index in full signal
        """
        start_idx = self.window_starts[idx]
        end_idx = min(start_idx + self.window_size, self.n_timepoints)

        target_window = self.target[:, start_idx:end_idx]

        return target_window, start_idx, end_idx


class FourierShiftDictionary(nn.Module):
    """
    Applies per-neuron time shifts using Fourier phase shifts.

    Time domain shifting is equivalent to phase shifts in frequency domain:
    shift(x(t), Δt) = IFFT(FFT(x(ω)) * exp(-i*ω*Δt))

    This is exact for any fractional shift and fully differentiable.

    Supports arbitrary shift ranges specified as [min_shift_ms, max_shift_ms].
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        max_shift_ms: float = 1000.0,
        sampling_rate: float = 14.0,
        init_shift_ms: Optional[torch.Tensor] = None
    ):
        """
        Args:
            dictionary: (n_neurons, n_timepoints) unshifted neural patterns
            max_shift_ms: Maximum allowed shift. Can be:
                         - float: shifts range from [0, max_shift_ms]
                         - list/tuple [min_shift_ms, max_shift_ms]: shifts range from [min_shift_ms, max_shift_ms]
            sampling_rate: Sampling frequency in Hz
            init_shift_ms: Optional initial shifts in ms (default: small random noise around midpoint of range)
        """
        super().__init__()

        n_neurons, n_timepoints = dictionary.shape
        self.n_neurons = n_neurons
        self.n_timepoints = n_timepoints
        self.sampling_rate = sampling_rate
        self.ms_per_sample = 1000.0 / sampling_rate

        # Parse shift range
        if isinstance(max_shift_ms, (list, tuple)):
            if len(max_shift_ms) != 2:
                raise ValueError(f"max_shift_ms must be a scalar or [min, max], got {max_shift_ms}")
            min_shift_ms, max_shift_ms = max_shift_ms
        else:
            # Backward compatibility: single value means [0, max_shift_ms]
            min_shift_ms = 0.0
            max_shift_ms = float(max_shift_ms)

        if min_shift_ms >= max_shift_ms:
            raise ValueError(f"min_shift_ms ({min_shift_ms}) must be less than max_shift_ms ({max_shift_ms})")

        # Convert to samples and store as buffers
        self.register_buffer('min_shift_samples',
                           torch.tensor(min_shift_ms / self.ms_per_sample, device=dictionary.device))
        self.register_buffer('max_shift_samples',
                           torch.tensor(max_shift_ms / self.ms_per_sample, device=dictionary.device))

        # Store shift range for convenience
        self.register_buffer('shift_range_samples',
                           self.max_shift_samples - self.min_shift_samples)
        self.register_buffer('shift_midpoint_samples',
                           (self.min_shift_samples + self.max_shift_samples) / 2)

        # Register unshifted dictionary as buffer (not trainable)
        self.register_buffer('dictionary_unshifted', dictionary)

        # Initialize shift parameters (in samples)
        if init_shift_ms is not None:
            init_shifts = torch.as_tensor(init_shift_ms, device=dictionary.device) / self.ms_per_sample
            if init_shifts.ndim == 0:  # scalar provided, broadcast to all neurons
                init_shifts = init_shifts.unsqueeze(0).expand(n_neurons).clone()
            elif init_shifts.shape[0] != n_neurons:
                raise ValueError(f"init_shift_ms must have length {n_neurons}, got {init_shifts.shape[0]}")

            # Validate initial shifts are within range
            if torch.any(init_shifts < self.min_shift_samples) or torch.any(init_shifts > self.max_shift_samples):
                raise ValueError(f"init_shift_ms must be within [{min_shift_ms}, {max_shift_ms}], "
                               f"got values in [{init_shifts.min().item() * self.ms_per_sample:.2f}, "
                               f"{init_shifts.max().item() * self.ms_per_sample:.2f}]")
        else:
            # Initialize with small random noise around midpoint to break symmetry (conservative: 2% of range)
            init_shifts = self.shift_midpoint_samples + torch.randn(n_neurons, device=dictionary.device) * (self.shift_range_samples * 0.02)
            # Clamp to ensure within bounds
            init_shifts = torch.clamp(init_shifts, self.min_shift_samples, self.max_shift_samples)

        # Store shifts as unconstrained parameters
        # We'll use sigmoid to constrain to [min_shift, max_shift]
        # Map [min_shift, max_shift] -> [0, 1] -> logit
        # normalized = (init_shifts - min_shift) / (max_shift - min_shift)
        normalized = (init_shifts - self.min_shift_samples) / self.shift_range_samples
        # Clamp to avoid numerical issues
        normalized = torch.clamp(normalized, 1e-6, 1 - 1e-6)
        # Inverse sigmoid: logit(p) = log(p / (1-p))
        self.shift_logits = nn.Parameter(
            torch.log(normalized / (1 - normalized))
        )

        # Precompute frequency vector for phase shifts
        # Ensure it's on the same device as dictionary
        freqs = torch.fft.rfftfreq(n_timepoints, device=dictionary.device)
        self.register_buffer('freqs', freqs)

    def get_shifts_samples(self) -> torch.Tensor:
        """Get current shift values in samples (constrained to [min_shift, max_shift])"""
        # sigmoid maps to [0, 1], then scale to [min_shift, max_shift]
        normalized = torch.sigmoid(self.shift_logits)
        return self.min_shift_samples + normalized * self.shift_range_samples

    def get_shifts_ms(self) -> torch.Tensor:
        """Get current shift values in milliseconds"""
        return self.get_shifts_samples() * self.ms_per_sample

    def forward(self) -> torch.Tensor:
        """
        Apply learned shifts to dictionary.

        Returns:
            shifted_dictionary: (n_neurons, n_timepoints)
        """
        shifts_samples = self.get_shifts_samples()

        # Vectorized Fourier shift: process all neurons in parallel
        # FFT of all neurons at once (batch operation along dim=1)
        # Input: (n_neurons, n_timepoints) -> Output: (n_neurons, n_freqs)
        fft_all = torch.fft.rfft(self.dictionary_unshifted, dim=1)

        # Phase shift for all neurons using broadcasting
        # shifts_samples: (n_neurons,) -> reshape to (n_neurons, 1)
        # self.freqs: (n_freqs,) -> broadcast to (n_neurons, n_freqs)
        # Result: (n_neurons, n_freqs)
        phase_shifts = torch.exp(-2j * np.pi * self.freqs.unsqueeze(0) * shifts_samples.unsqueeze(1))

        # Apply shifts (element-wise multiply)
        shifted_fft_all = fft_all * phase_shifts

        # Inverse FFT for all neurons at once (batch operation along dim=1)
        shifted_dictionary = torch.fft.irfft(shifted_fft_all, n=self.n_timepoints, dim=1)

        return shifted_dictionary
    
    
class SparseCodingWithShifts(nn.Module):
    """
    Sparse coding model with learnable shifts and exponential activation.

    Combines:
    - Learnable time shifts (via FourierShiftDictionary, Conv1DShiftDictionary, or GridSampleShiftDictionary)
    - Sparse code weights
    - Learnable exponential activation (mimicking DA burst dynamics)
    """

    def __init__(
        self,
        shift_dictionary: nn.Module,  # FourierShiftDictionary, Conv1DShiftDictionary, or GridSampleShiftDictionary
        n_neurons: int,
        init_code: Optional[torch.Tensor] = None,
        activation_type: str = 'global',
        activation_function: str = 'exponential'
    ):
        super().__init__()

        self.shift_dictionary = shift_dictionary
        self.n_neurons = n_neurons
        self.activation_type = activation_type
        self.activation_function = activation_function

        # Learnable sparse code
        if init_code is not None:
            self.code = nn.Parameter(init_code.clone())
        else:
            self.code = nn.Parameter(torch.randn(1, n_neurons) * 0.1)

        # Initialize activation based on type
        if activation_type == 'global':
            # Single activation applied after summation
            if activation_function == 'exponential':
                self.act = LearnableExponential(init_rate=1.0, init_scale=0.1)
            elif activation_function == 'sigmoid':
                self.act = LearnableSigmoid(init_Rmax=1.0, init_slope=4.0, init_c50=1.0)
            elif activation_function == 'tanh':
                self.act = nn.Tanh()
            else:
                raise ValueError(f"Unknown activation_function: {activation_function}")
        elif activation_type == 'per_neuron':
            # Separate activation for each neuron/dictionary element
            if activation_function == 'exponential':
                self.activations = nn.ModuleList([
                    LearnableExponential(init_rate=1.0, init_scale=0.1, single_sided=True)
                    for _ in range(n_neurons)
                ])
            elif activation_function == 'sigmoid':
                self.activations = nn.ModuleList([
                    LearnableSigmoid(init_Rmax=1.0, init_slope=8.0, init_c50=1.0, single_sided=True)
                    for _ in range(n_neurons)
                ])
            elif activation_function == 'tanh':
                self.activations = nn.ModuleList([
                    nn.Tanh()
                    for _ in range(n_neurons)
                ])
            else:
                raise ValueError(f"Unknown activation_function: {activation_function}")
        elif activation_type == 'legacy':
            # Old TwoExpFastEnd for compatibility
            self.act = TwoExpFastEnd()
        else:
            raise ValueError(f"Unknown activation_type: {activation_type}")

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable activation (for backward compatibility with legacy mode)"""
        if self.activation_type == 'legacy':
            # Legacy exponential activation (using TwoExpFastEnd)
            x_pos = x.clamp(min=0.0)
            x_neg = (-x).clamp(min=0.0)
            pos_part = self.act.A_pos * torch.exp(x_pos * self.act.alpha)
            neg_part = -self.act.A_neg * torch.exp(x_neg * self.act.alpha)
            return (pos_part + neg_part).clamp(-1, 1)
        else:
            # Use the act module for global activation
            return self.act(x)

    def forward(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass: get shifted dictionary, multiply by code, apply activation.

        Args:
            start_idx: Optional start index for windowing the dictionary (default: None, use full dictionary)
            end_idx: Optional end index for windowing the dictionary (default: None, use full dictionary)

        Returns:
            reconstruction: (1, n_timepoints) or (1, window_size) if windowed
        """
        # Get shifted dictionary
        dictionary_shifted = self.shift_dictionary()

        # Apply windowing if indices provided
        if start_idx is not None and end_idx is not None:
            dictionary_shifted = dictionary_shifted[:, start_idx:end_idx]

        if self.activation_type == 'global' or self.activation_type == 'legacy':
            # Linear reconstruction then activation
            linear_recon = self.code @ dictionary_shifted
            reconstruction = self.activation(linear_recon)

        elif self.activation_type == 'per_neuron':
            # Apply per-neuron activation to dictionary first
            activated_dictionary = self._batch_activate_per_neuron(dictionary_shifted)
            reconstruction = self.code @ activated_dictionary

        return reconstruction

    def _batch_activate_per_neuron(self, neuron_contributions: torch.Tensor) -> torch.Tensor:
        """
        Efficiently apply per-neuron activations in batch.

        Parameters:
        -----------
        neuron_contributions : torch.Tensor
            Shape (n_components, n_features) - contributions from each neuron

        Returns:
        --------
        torch.Tensor
            Shape (n_components, n_features) - activated contributions
        """
        if isinstance(self.activations[0], LearnableExponential):
            return self._batch_activate_per_neuron_exponential(neuron_contributions)
        elif isinstance(self.activations[0], LearnableSigmoid):
            return self._batch_activate_per_neuron_sigmoid(neuron_contributions)
        else:
            raise NotImplementedError(f"Batch activation not implemented for {type(self.activations[0])}")

    def _batch_activate_per_neuron_exponential(self, neuron_contributions: torch.Tensor) -> torch.Tensor:
        """Batch apply exponential activations."""
        rates_pos = torch.stack([act.rate_pos for act in self.activations])
        scales_pos = torch.stack([act.scale_pos for act in self.activations])

        if self.activations[0].single_sided:
            return scales_pos.unsqueeze(-1) * (torch.exp(rates_pos.unsqueeze(-1) * neuron_contributions) - 1)

        if self.activations[0].separate_pos_neg:
            rates_neg = torch.stack([act.rate_neg for act in self.activations])
            scales_neg = torch.stack([act.scale_neg for act in self.activations])

        x_pos = neuron_contributions.clamp(min=0.0)
        x_neg = (-neuron_contributions).clamp(min=0.0)

        y_pos = scales_pos.unsqueeze(-1) * (torch.exp(rates_pos.unsqueeze(-1) * x_pos) - 1)

        if self.activations[0].separate_pos_neg:
            y_neg = -scales_neg.unsqueeze(-1) * (torch.exp(rates_neg.unsqueeze(-1) * x_neg) - 1)
        else:
            y_neg = -scales_pos.unsqueeze(-1) * (torch.exp(rates_pos.unsqueeze(-1) * x_neg) - 1)

        return y_pos + y_neg

    def _batch_activate_per_neuron_sigmoid(self, neuron_contributions: torch.Tensor) -> torch.Tensor:
        """Batch apply sigmoid activations."""
        Rmax_pos = torch.stack([act.Rmax_pos for act in self.activations])
        slope_pos = torch.stack([act.slope_pos for act in self.activations])
        c50_pos = torch.stack([act.c50_pos for act in self.activations])

        if self.activations[0].single_sided:
            return Rmax_pos.unsqueeze(-1) / (
                1 + torch.exp(-slope_pos.unsqueeze(-1) * (neuron_contributions - c50_pos.unsqueeze(-1)))
            )

        Rmax_neg = torch.stack([act.Rmax_neg for act in self.activations])
        slope_neg = torch.stack([act.slope_neg for act in self.activations])
        c50_neg = torch.stack([act.c50_neg for act in self.activations])

        x_pos = neuron_contributions.clamp(min=0.0)
        x_neg = (-neuron_contributions).clamp(min=0.0)

        y_pos = Rmax_pos.unsqueeze(-1) / (
            1 + torch.exp(-slope_pos.unsqueeze(-1) * (x_pos - c50_pos.unsqueeze(-1)))
        )

        y_neg = -Rmax_neg.unsqueeze(-1) / (
            1 + torch.exp(-slope_neg.unsqueeze(-1) * (x_neg - c50_neg.unsqueeze(-1)))
        )

        return y_pos + y_neg

    def get_code(self) -> torch.Tensor:
        """Get the raw code."""
        return self.code

    def get_activation_params(self) -> dict:
        """Get learned activation parameters for analysis."""
        if self.activation_type == 'per_neuron':
            if isinstance(self.activations[0], LearnableExponential):
                params = {
                    'rates_pos': [act.rate_pos.item() for act in self.activations],
                    'scales_pos': [act.scale_pos.item() for act in self.activations],
                }
                if self.activations[0].separate_pos_neg:
                    params['rates_neg'] = [act.rate_neg.item() for act in self.activations]
                    params['scales_neg'] = [act.scale_neg.item() for act in self.activations]
            elif isinstance(self.activations[0], LearnableSigmoid):
                params = {
                    'Rmax_pos': [act.Rmax_pos.item() for act in self.activations],
                    'slope_pos': [act.slope_pos.item() for act in self.activations],
                    'c50_pos': [act.c50_pos.item() for act in self.activations],
                }
                if self.activations[0].separate_pos_neg:
                    params['Rmax_neg'] = [act.Rmax_neg.item() for act in self.activations]
                    params['slope_neg'] = [act.slope_neg.item() for act in self.activations]
                    params['c50_neg'] = [act.c50_neg.item() for act in self.activations]
            else:
                params = {}
        elif self.activation_type == 'global':
            if isinstance(self.act, LearnableExponential):
                params = {
                    'rate_pos': self.act.rate_pos.item(),
                    'scale_pos': self.act.scale_pos.item(),
                }
                if self.act.separate_pos_neg:
                    params['rate_neg'] = self.act.rate_neg.item()
                    params['scale_neg'] = self.act.scale_neg.item()
            elif isinstance(self.act, LearnableSigmoid):
                params = {
                    'Rmax_pos': self.act.Rmax_pos.item(),
                    'slope_pos': self.act.slope_pos.item(),
                    'c50_pos': self.act.c50_pos.item(),
                }
                if self.act.separate_pos_neg:
                    params['Rmax_neg'] = self.act.Rmax_neg.item()
                    params['slope_neg'] = self.act.slope_neg.item()
                    params['c50_neg'] = self.act.c50_neg.item()
            else:
                params = {}
        else:
            params = {}
        return params


def normal_twoend(x):
    """Normalize data to range [-1, 1]"""
    x = 2 * (x - x.min()) / (x.max() - x.min()) - 1  # normalize to [-1, 1]
    return x
    
def prepare_data_for_encoding(xr_session, signal2analyze, shuffle_trials=False):
    """
    Prepare atoms and target data for sparse encoding.
    
    Parameters
    ----------
    xr_session : xarray.Dataset
        Session data containing spikes, photometry, and trial information
    signal2analyze : str
        Name of the signal variable to analyze (e.g., 'zscored_df_over_f')
    mask_idx : array-like
        Boolean mask for valid trials
    shuffle_trials: bool
        whether to shuffle the trials before training the model
        
    Returns
    -------
    dict
        Dictionary containing prepared data including:
        - dict_atoms_smooth: normalized and smoothed dictionary atoms
        - target_stack_smooth: normalized and smoothed target signal
        - event_time: time coordinates
        - cluID_atom: cluster IDs
        - lick_rate: lick rate data
    """
    atoms = xr_session['spikes_FR_session'].data  # original dimen: trial x time x cluID
    atoms = atoms.transpose([1, 2, 0])  # time x cluID x trial

    target = xr_session[signal2analyze].data
    target = target.T  # time x trial
    lick_rate = xr_session['lick_rate'].data #not used for now
    lick_rate = lick_rate.T

    mask_idx = (~np.isnan(atoms[0, 0, :])) & (~np.isnan(target[0, :]))


    # Filter valid trials
    atoms = atoms[:, :, mask_idx]
    target = target[:, mask_idx]
    lick_rate = lick_rate[:, mask_idx]
    event_time = xr_session.time
    cluID_atom = xr_session.cluID.data
    
    # smoothing
    atoms_smooth = savgol_filter(atoms, 21,2, axis=0)
    target_smooth = savgol_filter(target, 21,2, axis=0)


    # Concentate trials together
    atoms_stack = atoms_smooth.transpose([1, 2, 0])
    atoms_stack = atoms_stack.reshape(atoms_stack.shape[0], -1) # concatenate trials together


    # shuffling if required
    if shuffle_trials:
        logger.info('Shuffling trials for target')
        n_trials = target.shape[1]
        shuffled_trial_indices = np.random.permutation(n_trials)
        target = target[:, shuffled_trial_indices]  

    target_stack_smooth = target_smooth.T.reshape(1, -1)
    target_stack_smooth = normal_twoend(target_stack_smooth)
        
    # Add intercept baseline
    atoms_stack_norm = normalize(atoms_stack, axis=1)
    baseline_atom = -np.ones((1, atoms_stack_norm.shape[1]))
    dict_atoms_smooth = np.vstack([atoms_stack_norm, baseline_atom])
    
    # Reshape dict_atoms_smooth back to time x cluID x trial
    n_cluID = atoms.shape[1]
    n_trials = atoms.shape[2]
    n_timepoints = atoms.shape[0]
    # Keep all atoms including baseline (last row) before reshaping
    # Note: n_cluID + 1 accounts for the baseline atom
    atoms_norm_smooth = atoms_stack_norm.reshape(n_cluID, n_trials, n_timepoints).transpose([2, 0, 1])

    # Reshape target_stack_smooth back to time x trial
    target_norm_smooth = target_stack_smooth.reshape(n_trials, n_timepoints).T
    return {
        'dict_atoms_smooth': dict_atoms_smooth, #all the trials already concatenate together
        'target_stack_smooth': target_stack_smooth,
        'atoms': atoms,
        'target': target,
        'atoms_smooth': atoms_smooth,
        'atoms_stack': atoms_stack,
        'atoms_norm_smooth': atoms_norm_smooth,  # time x cluID x trial
        'target_norm_smooth': target_norm_smooth,  # time x trial
        'mask_idx': mask_idx,
        'event_time': event_time,
        'cluID_atom': cluID_atom,
        'trial_outcome': xr_session.trial_outcome[mask_idx],
        'trial_nb': xr_session.trial_nb[mask_idx],
        'lick_rate': lick_rate
    }


def prepare_fold_from_atoms_target(
    atoms: np.ndarray,
    target: np.ndarray,
    trial_indices: np.ndarray,
    lick_rate: np.ndarray = None
) -> dict:
    """
    Prepare stacked and normalized data for a subset of trials (for cross-validation).

    This function replicates the core logic of prepare_data_for_encoding() but works on
    already-filtered trial subsets. Used for creating train/test splits in CV.

    Parameters
    ----------
    atoms : np.ndarray
        Neural data, shape (time × cluID × n_trials) - already filtered for NaN
    target : np.ndarray
        Photometry data, shape (time × n_trials) - already filtered for NaN
    trial_indices : np.ndarray
        Indices of trials to include in this fold
    lick_rate : np.ndarray, optional
        Lick rate data, shape (time × n_trials)

    Returns
    -------
    dict
        Dictionary with keys:
        - dict_atoms_smooth: smoothed dictionary atoms, shape (n_atoms × time*n_fold_trials)
        - target_stack_smooth: smoothed target, shape (1 × time*n_fold_trials)
        - target_stack: raw target stack, shape (1 × time*n_fold_trials)
        - n_trials: number of trials in this fold
    """
    # Subset trials
    atoms_subset = atoms[:, :, trial_indices]  # (time × cluID × n_fold_trials)
    target_subset = target[:, trial_indices]  # (time × n_fold_trials)

    # Stack atoms
    atoms_stack = atoms_subset.transpose([1, 2, 0])  # (cluID × n_fold_trials × time)
    atoms_stack = atoms_stack.reshape(atoms_stack.shape[0], -1)  # (cluID × [time*n_fold_trials])

    # Need to add back the baseline
    baseline_atom = -np.ones((1, atoms_stack.shape[1]))
    dict_atoms = np.vstack([atoms_stack, baseline_atom])

    # Stack and normalize target (same as lines 2303-2304)
    target_stack = target_subset.T.reshape(1, -1)  # (1 × [time*n_fold_trials])

    return {
        'dict_atoms_smooth': dict_atoms,
        'target_stack_smooth': target_stack,
        'n_trials': len(trial_indices)
    }


def prepare_data_for_encoding_trials(
    xr_session,
    signal2analyze,
    shuffle_trials=False
) -> dict:
    """
    Prepare trial-stacked data for sparse encoding (wrapper around prepare_data_for_encoding).

    This function returns the same normalized data as prepare_data_for_encoding() but in
    trial-stacked format suitable for the new sparse_encode_pytorch_with_shift_trials()
    wrapper function. This enables cleaner cross-validation without manual concatenation.

    IMPORTANT: This function reshapes the CONCATENATED and NORMALIZED dict_atoms_smooth
    back to trial format to preserve the exact same normalization statistics as the
    original concatenated approach.

    Parameters
    ----------
    xr_session : xarray.Dataset
        Session data containing spikes, photometry, and trial information
    signal2analyze : str
        Name of the signal variable to analyze (e.g., 'dff_znorm', 'zscored_df_over_f')
    shuffle_trials : bool
        Whether to shuffle trials before training (default: False)

    Returns
    -------
    dict
        Dictionary containing prepared data:
        - 'atoms_norm_smooth': Normalized atoms WITH baseline, shape (time, n_neurons+1, n_trials)
        - 'target_norm_smooth': Normalized target, shape (time, n_trials)
        - 'lick_rate': Lick rate data, shape (time, n_trials)
        - 'trial_outcome': Trial outcomes (filtered for valid trials)
        - 'trial_nb': Trial numbers (filtered for valid trials)
        - 'n_neurons': Number of neurons including baseline
        - 'n_timepoints': Number of timepoints per trial
        - 'n_trials': Number of valid trials
        - 'event_time': Time coordinates
        - 'cluID_atom': Cluster IDs

    Examples
    --------
    >>> full_data = prepare_data_for_encoding_trials(xr_session, 'dff_znorm')
    >>> atoms = full_data['atoms_norm_smooth']  # (time, n_neurons+1, n_trials)
    >>> target = full_data['target_norm_smooth']  # (time, n_trials)
    >>> dict_atoms = atoms.transpose([1, 0, 2])  # (n_neurons+1, time, n_trials) for model

    Notes
    -----
    - This function calls the existing prepare_data_for_encoding() internally
    - Uses dict_atoms_smooth (already normalized and with baseline) and reshapes it
    - Preserves exact normalization statistics from concatenated approach
    - Output format is optimized for trial-wise indexing in cross-validation
    """
    # Call existing function to get normalized data
    full_data = prepare_data_for_encoding(
        xr_session=xr_session,
        signal2analyze=signal2analyze,
        shuffle_trials=shuffle_trials
    )

    # Extract the ALREADY NORMALIZED and CONCATENATED dict_atoms_smooth
    # This includes baseline and has shape (n_neurons+1, time*n_trials)
    dict_atoms_concat = full_data['dict_atoms_smooth']  # (n_neurons+1, time*n_trials)
    target_concat = full_data['target_stack_smooth']      # (1, time*n_trials)

    # Get dimensions
    n_neurons_with_baseline = dict_atoms_concat.shape[0]  # includes baseline
    n_timepoints = full_data['atoms'].shape[0]
    n_trials = full_data['atoms'].shape[2]

    # Reshape dict_atoms from (n_neurons+1, time*n_trials) back to (n_neurons+1, n_trials, time)
    # then transpose to (time, n_neurons+1, n_trials)
    dict_atoms_trial = dict_atoms_concat.reshape(n_neurons_with_baseline, n_trials, n_timepoints)
    dict_atoms_trial = dict_atoms_trial.transpose([2, 0, 1])  # (time, n_neurons+1, n_trials)

    # Reshape target from (1, time*n_trials) to (time, n_trials)
    target_trial = target_concat.reshape(n_trials, n_timepoints).T  # (time, n_trials)

    return {
        'atoms_norm_smooth': dict_atoms_trial,   # (time, n_neurons+1, n_trials) - includes baseline, properly normalized
        'target_norm_smooth': target_trial,      # (time, n_trials) - properly normalized
        'lick_rate': full_data['lick_rate'],     # (time, n_trials)
        'trial_outcome': full_data['trial_outcome'],
        'trial_nb': full_data['trial_nb'],
        'n_neurons': n_neurons_with_baseline,  # Include baseline
        'n_timepoints': n_timepoints,
        'n_trials': n_trials,
        'event_time': full_data['event_time'],
        'cluID_atom': full_data['cluID_atom']
    }


def plot_reconstruction_comparison(target_stack_smooth, reconstruction, time_range=(100, 150), sampling_rate=50):
    """
    Plot comparison between target signal and reconstruction.
    
    Parameters
    ----------
    target_stack_smooth : np.ndarray
        Smoothed target signal
    reconstruction : np.ndarray
        Reconstructed signal from sparse encoding
    time_range : tuple, optional
        Time range to display (start, end) in seconds
    sampling_rate : float, optional
        Sampling rate in Hz
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    t = np.arange(len(reconstruction.ravel())) / sampling_rate
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(t, target_stack_smooth.T, label='Target')
    ax.plot(t, reconstruction.T, 'r', label='Reconstruction')
    ax.set_xlim(time_range)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal')
    ax.legend()
    return fig, ax


def evaluate_sparse_model_on_test_data(
    model_train,
    dict_test: np.ndarray,
    target_test: np.ndarray,
    sampling_rate: float,
    max_shift_ms: list = [-200, 200],
    device: str = 'cuda'
) -> dict:
    """
    Evaluate a trained sparse encoding model on test data.

    This function applies a trained sparse encoding model (with learned shifts and codes)
    to test data by freezing all learned parameters and computing predictions without
    any gradient updates or optimization.

    Parameters
    ----------
    model_train : SparseCodingWithShifts
        Trained model with learned parameters (shifts, codes, activation function)
    dict_test : np.ndarray
        Test dictionary atoms with shape (n_neurons, time, n_trials)
    target_test : np.ndarray
        Test target signal with shape (time, n_trials)
    sampling_rate : float
        Sampling rate in Hz
    max_shift_ms : list, optional
        Maximum shift range in milliseconds [min, max]. Default is [-200, 200]
    device : str, optional
        Device to run computation on ('cuda' or 'cpu'). Default is 'cuda'

    Returns
    -------
    dict
        Dictionary containing:
        - 'reconstruction': reconstructed test signal (time, n_trials)
        - 'test_r2': R² score on test data
        - 'test_mse': mean squared error on test data
        - 'test_var': variance of test target
        - 'model_test': the test model with frozen parameters

    Notes
    -----
    - The function creates a new model instance for test data but copies all learned
      parameters from the training model (shifts, codes, activation function state)
    - No gradients are computed and no optimization is performed
    - The test model uses the same architecture as the training model

    Examples
    --------
    >>> results = evaluate_sparse_model_on_test_data(
    ...     model_train=trained_model,
    ...     dict_test=test_atoms,
    ...     target_test=test_target,
    ...     sampling_rate=50.0
    ... )
    >>> print(f"Test R²: {results['test_r2']:.4f}")
    """
    # Get dimensions
    n_neurons_test, n_timepoints_test, n_trials_test = dict_test.shape

    # Flatten test data to match training format
    # (n_neurons, time, trials) → (n_neurons, trials, time) → (n_neurons, trials*time)
    dict_test_flat = dict_test.transpose([0, 2, 1])  # (n_neurons, trials, time)
    dict_test_flat = dict_test_flat.reshape(n_neurons_test, n_trials_test * n_timepoints_test)

    # Target: (time, trials) → (trials, time) → flatten
    target_test_flat = target_test.T.reshape(1, n_trials_test * n_timepoints_test)

    with torch.no_grad():
        model_train.eval()

        # Apply learned model to flattened test dictionary
        test_dict_tensor = torch.FloatTensor(dict_test_flat).to(device)

        # Create new FourierShiftDictionary for test data
        test_shift_dict = FourierShiftDictionary(
            dictionary=test_dict_tensor,
            max_shift_ms=max_shift_ms,
            sampling_rate=sampling_rate
        ).to(device)

        # Copy learned shifts from training
        test_shift_dict.shift_logits.data = model_train.shift_dictionary.shift_logits.data

        # Create test model
        model_test = SparseCodingWithShifts(
            shift_dictionary=test_shift_dict,
            n_neurons=test_dict_tensor.shape[0],
            activation_type='global'  # Match training
        ).to(device)

        # Copy learned code and activation from training
        model_test.code.data = model_train.code.data
        model_test.act.load_state_dict(model_train.act.state_dict())

        # Forward pass (no gradients, no optimization!)
        reconstruction_test_flat = model_test().cpu().numpy()

        # Reshape back to (time, trials)
        reconstruction_test = reconstruction_test_flat.reshape(n_trials_test, n_timepoints_test).T

        # Calculate test R²
        test_mse = np.mean((reconstruction_test - target_test) ** 2)
        test_var = np.var(target_test)
        test_r2 = 1 - (test_mse / test_var)

    return {
        'reconstruction': reconstruction_test,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_var': test_var,
        'model_test': model_test
    }