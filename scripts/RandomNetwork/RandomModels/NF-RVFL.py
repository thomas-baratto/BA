import typing
import torch
import numpy as np
from sklearn.cluster import KMeans
from .utils import torch_activation, ensure_tensor

# Attempt to import scikit-fuzzy for Fuzzy C-Means (NF-RVFL-C)
try:
    import skfuzzy as fuzz
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False

class NF_RVFL:
    """Neuro-Fuzzy Random Vector Functional Link Network (NF-RVFL).
    
    Integrates fuzzy logic into the RVFL framework.
    Supported variations:
     - "R" (NF-RVFL-R): Randomly initialized clustering centers
     - "K" (NF-RVFL-K): K-Means clustering centers
     - "C" (NF-RVFL-C): Fuzzy C-Means clustering centers (requires scikit-fuzzy)
     
    Outputs are determined by concatenated [Fuzzified Features, Hidden Node Activations, Original Input].
    """

    def __init__(self,
                 n_hidden: int = 100,
                 n_rules: int = 10, # Number of fuzzy rules / clusters
                 variation: str = "K", # "R", "K", or "C"
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_hidden = int(n_hidden)
        self.n_rules = int(n_rules)
        self.variation = variation.upper()
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        if self.variation == "C" and not HAS_SKFUZZY:
            raise ImportError("scikit-fuzzy must be installed to use Fuzzy C-Means (NF-RVFL-C). Use variation='K' or 'R' instead.")

        self.W_hidden: typing.Optional[torch.Tensor] = None
        self.b_hidden: typing.Optional[torch.Tensor] = None
        self.W_out: typing.Optional[torch.Tensor] = None
        
        # Fuzzy layer parameters
        self.centers: typing.Optional[torch.Tensor] = None
        self.widths: typing.Optional[torch.Tensor] = None

    def _init_weights(self, n_features: int):
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))
            
        # The hidden layer maps the fuzzified features to random nodes
        # Fuzzified features dimension = n_rules
        self.W_hidden = torch.randn(self.n_rules, self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
        if self.include_bias:
            self.b_hidden = torch.randn(self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
        else:
            self.b_hidden = torch.zeros(self.n_hidden, dtype=torch.float64, device=self.device)



    def _fuzzify(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate membership grades (fuzzification) using Gaussian membership functions."""
        # X shape: (N, D), centers shape: (n_rules, D), widths shape: (n_rules, D)
        # We compute Gaussian distances for each sample to each cluster center
        N = X.shape[0]
        # Expand dims for broadcasting: X->(N, 1, D)
        X_exp = X.unsqueeze(1)
        
        # dists shape: (N, n_rules, D)
        dists = (X_exp - self.centers.unsqueeze(0)) ** 2
        
        # widths_exp shape: (1, n_rules, D)
        widths_exp = (self.widths.unsqueeze(0) + 1e-8) # avoid div zero
        
        # Gaussians per feature per rule: exp(- (x - center)^2 / width^2)
        gauss_vals = torch.exp(-dists / (2 * widths_exp))
        
        # Multiply across features to get rule activation (T-norm: product)
        # Shape: (N, n_rules)
        rule_activations = torch.prod(gauss_vals, dim=2)
        
        # Normalize rule activations
        sum_activations = torch.sum(rule_activations, dim=1, keepdim=True) + 1e-8
        normalized_activations = rule_activations / sum_activations
        
        return normalized_activations

    def _determine_fuzzy_parameters(self, X_t: torch.Tensor):
        X_np = X_t.cpu().numpy()
        N, D = X_np.shape
        
        if self.variation == "K":
            # K-Means
            kmeans = KMeans(n_clusters=self.n_rules, random_state=self.random_state, n_init=10)
            kmeans.fit(X_np)
            centers = kmeans.cluster_centers_
            
            # calculate widths using nearest neighbor heuristic
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(centers, centers)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            # Use distance to nearest cluster center as width parameter for all features in that center
            widths = np.tile((min_dists / np.sqrt(2))[:, np.newaxis], (1, D))
            
        elif self.variation == "C":
            # Fuzzy C-Means (requires scikit-fuzzy, cmeans expects X.T)
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X_np.T, self.n_rules, 2, error=0.005, maxiter=1000, seed=self.random_state
            )
            centers = cntr
            # Same heuristic for widths
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(centers, centers)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            widths = np.tile((min_dists / np.sqrt(2))[:, np.newaxis], (1, D))
            
        else: # "R" (Randomly initialized centers)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(N, self.n_rules, replace=False)
            centers = X_np[indices]
            
            # Use global variance for widths
            global_var = np.var(X_np, axis=0)
            widths = np.tile(global_var[np.newaxis, :], (self.n_rules, 1))

        self.centers = torch.from_numpy(centers).to(dtype=torch.float64, device=self.device)
        self.widths = torch.from_numpy(widths).to(dtype=torch.float64, device=self.device)

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        X_t = ensure_tensor(X, self.device)
        N, D = X_t.shape
        
        if self.centers is None:
            self._determine_fuzzy_parameters(X_t)
            
        if self.W_hidden is None:
            self._init_weights(D)

        if isinstance(y, np.ndarray):
            Y_t = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)
        else:
            Y_t = y.to(dtype=torch.float64, device=self.device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(1)

        # 1. Fuzzification
        Z_fuzzy = self._fuzzify(X_t) # (N, n_rules)
        
        # 2. Random Hidden Layer over Fuzzified features
        H_random = Z_fuzzy.matmul(self.W_hidden) + self.b_hidden
        H_random = torch_activation(H_random, self.activation) # (N, n_hidden)
        
        # 3. Defuzzification mapping interface - Combine Z_fuzzy, H_random, and original input X
        H_final = torch.cat([Z_fuzzy, H_random, X_t], dim=1)
        
        # Solve Ridge Regression: W_out = (H_final^T H_final + alpha I)^(-1) H_final^T Y
        L = H_final.shape[1]
        A = H_final.T.matmul(H_final) + self.alpha * torch.eye(L, dtype=torch.float64, device=self.device)
        B = H_final.T.matmul(Y_t)
        self.W_out = torch.linalg.solve(A, B)

        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("The model hasn't been fitted. Call fit() first.")
            
        X_t = ensure_tensor(X, self.device)
        
        Z_fuzzy = self._fuzzify(X_t)
        H_random = Z_fuzzy.matmul(self.W_hidden) + self.b_hidden
        H_random = torch_activation(H_random, self.activation)
        
        H_final = torch.cat([Z_fuzzy, H_random, X_t], dim=1)
        
        scores = H_final.matmul(self.W_out)
        
        scores_np = scores.cpu().numpy()
        if scores_np.shape[1] == 1:
            return scores_np.ravel()
        return scores_np