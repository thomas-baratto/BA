import typing
import torch
import numpy as np
from .utils import ensure_tensor

class FELM:
    """Functional Extreme Learning Machine (FELM).
    
    Instead of using standard activation functions with random weights and biases, 
    FELM employs functional neurons (e.g., using Polynomial or Fourier expansion bases). 
    The network maps the input to a functional basis space, and analytically solves 
    for the coefficients of these bases to achieve high-precision fitting.
    """

    def __init__(self,
                 n_basis: int = 5,
                 basis_type: str = "polynomial", # "polynomial" or "fourier"
                 alpha: float = 1e-3,
                 device: typing.Optional[torch.device] = None):
        """
        Args:
            n_basis (int): The number of expansion terms (e.g. polynomial degree).
            basis_type (str): Type of functional basis ("polynomial" or "fourier").
            alpha (float): Ridge regression regularization parameter.
            device (torch.device): Compute device.
        """
        self.n_basis = int(n_basis)
        self.basis_type = basis_type.lower()
        self.alpha = float(alpha)
        self.device = device if device is not None else torch.device("cpu")

        self.W_out: typing.Optional[torch.Tensor] = None

    def _expand_basis(self, X: torch.Tensor) -> torch.Tensor:
        """Map the input features X into the specified functional basis space.
        
        Args:
            X (torch.Tensor): Shape (N, D)
            
        Returns:
            torch.Tensor: Shape (N, D * n_basis) or (N, D + D * n_basis * 2) depending on type.
        """
        N, D = X.shape
        
        if self.basis_type == "polynomial":
            # Polynomial expansion: [X, X^2, X^3, ..., X^n_basis]
            # Output shape: (N, D * n_basis)
            expanded = []
            for degree in range(1, self.n_basis + 1):
                expanded.append(X ** degree)
            return torch.cat(expanded, dim=1)
            
        elif self.basis_type == "fourier":
            # Fourier expansion: [X, sin(X), cos(X), sin(2X), cos(2X), ...]
            # Output shape: (N, D + D * n_basis * 2)
            expanded = [X]
            for freq in range(1, self.n_basis + 1):
                expanded.append(torch.sin(freq * X))
                expanded.append(torch.cos(freq * X))
            return torch.cat(expanded, dim=1)
            
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the FELM by analytically solving for the basis coefficients."""
        X_t = ensure_tensor(X, self.device)
        
        # Prepare Y
        if isinstance(y, np.ndarray):
            Y_t = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)
        else:
            Y_t = y.to(dtype=torch.float64, device=self.device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(1)

        # Map to functional basis
        H = self._expand_basis(X_t)
        
        # Analytically solve for coefficients using Ridge Regression
        # W_out = (H^T H + alpha I)^(-1) H^T Y
        L = H.shape[1]
        A = H.T.matmul(H) + self.alpha * torch.eye(L, dtype=torch.float64, device=self.device)
        B = H.T.matmul(Y_t)
        self.W_out = torch.linalg.solve(A, B)

        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values using the learned functional coefficients."""
        if self.W_out is None:
            raise RuntimeError("The model hasn't been fitted. Call fit() first.")
            
        X_t = ensure_tensor(X, self.device)
        H = self._expand_basis(X_t)
        scores = H.matmul(self.W_out)
        
        scores_np = scores.cpu().numpy()
        if scores_np.shape[1] == 1:
            return scores_np.ravel()
        return scores_np
