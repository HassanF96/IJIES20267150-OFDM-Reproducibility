"""Fixed, source-checked HELENA adaptation and symbol-block Huber MAP baseline.

Receiver inputs only. Published performance and training data are not imported.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import gamma
from src.ofdm.information import require_pilots
from src.estimators.practical import design, masked_output
from src.estimators.noise import estimate_variance
from src.utils.cost import count


def same_conv(x, layer):
    kh, kw = layer.kernel_size
    # Keras SAME: odd extra padding goes to bottom/right.
    return layer(F.pad(x, ((kw-1)//2, kw//2, (kh-1)//2, kh//2)))


def patchify(x):
    """NCHW -> [batch, frequency patches, p*time*channels], NHWC order."""
    b, c, n, t = x.shape
    if n % 12: raise ValueError('Frequency dimension must divide into patches12')
    return x.permute(0, 2, 3, 1).contiguous().reshape(b, n//12, 12*t*c)


def unpatchify(x, symbols):
    b, p, d = x.shape
    if d != 12*symbols*2: raise ValueError('Reconstruction patch dimension')
    return x.reshape(b, p*12, symbols, 2).permute(0, 3, 1, 2).contiguous()


class Helena(nn.Module):
    """Section III / author serialized config; 96x8 local grid adaptation."""
    def __init__(self, symbols=8):
        super().__init__(); self.symbols = symbols
        self.conv1 = nn.Conv2d(2, 32, (12, 2))
        self.conv2 = nn.Conv2d(32, 2, (6, 7))
        self.drop = nn.Dropout(.1)
        self.embed = nn.Linear(12*symbols*2, 64)
        self.q = nn.Linear(64, 256); self.k = nn.Linear(64, 256); self.v = nn.Linear(64, 256)
        self.projection = nn.Linear(256, 64)
        self.norm = nn.LayerNorm(64, eps=.001)
        self.se1 = nn.Linear(64, 16); self.se2 = nn.Linear(16, 64)
        self.reconstruction = nn.Linear(64, 12*symbols*2)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        # Keras EinsumDense Glorot fans for [input, heads, key_dim] and
        # [heads, value_dim, output], unlike a flattened Linear initializer.
        for layer in (self.q, self.k, self.v):
            nn.init.uniform_(layer.weight, -math.sqrt(6/(256+4096)), math.sqrt(6/(256+4096)))
        nn.init.uniform_(self.projection.weight, -math.sqrt(6/512), math.sqrt(6/512))

    def attention(self, z):
        b, p, _ = z.shape
        def project(layer): return layer(z).reshape(b, p, 4, 64).transpose(1, 2)
        q, k, v = project(self.q), project(self.k), project(self.v)
        a = torch.softmax((q @ k.transpose(-1, -2))/8, dim=-1)
        h = (a @ v).transpose(1, 2).contiguous().reshape(b, p, 256)
        return self.projection(h)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 2 or x.shape[3] != self.symbols:
            raise ValueError('HELENA input [B,2,active_frequency,symbols]')
        f = self.drop(F.relu(same_conv(F.relu(same_conv(x, self.conv1)), self.conv2)))
        z = self.embed(patchify(f)); z = self.norm(z+self.attention(z))
        excitation = torch.sigmoid(self.se2(F.relu(self.se1(z.mean(1)))))
        return x + unpatchify(self.reconstruction(z*excitation[:, None]), self.symbols)


def helena_input(pilots):
    require_pilots(pilots); grid = pilots.grid
    sparse = np.zeros(grid.shape, complex)
    sparse[pilots.fit_mask] = pilots.y_allowed[pilots.fit_mask]/grid.pilot_grid[pilots.fit_mask]
    sparse = sparse[grid.active_bins]
    return np.stack((sparse.real, sparse.imag)).astype(np.float32)


def helena_estimate(pilots, model, device='cpu', ledger=None):
    x = helena_input(pilots)
    with torch.inference_mode(): y = model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
    out = np.zeros(pilots.grid.shape, complex)
    out[pilots.grid.active_bins] = y[0].astype(float)+1j*y[1].astype(float)
    if not np.isfinite(out).all(): raise ArithmeticError('HELENA nonfinite prediction')
    count(ledger, 'network_passes'); count(ledger, 'full_fits')
    return out


def block_huber(pilots, stats, iterations=5, ledger=None, details=False):
    """Symbol-group radial-Huber MAP, no independence guarantee under bursts."""
    basis, a, y = design(pilots, stats.support)
    root = np.linalg.cholesky(stats.covariance)
    b = a @ root; target = y-a @ stats.mean
    variance = estimate_variance(pilots, 1e-10)+1e-8
    symbols = np.broadcast_to(np.arange(pilots.grid.symbols), pilots.grid.shape)[pilots.fit_mask]
    groups = [np.flatnonzero(symbols == ell) for ell in range(pilots.grid.symbols)]
    if any(len(g) == 0 for g in groups): raise ValueError('Symbol-block fit needs pilots in every symbol')
    kappas = np.asarray([np.sqrt(gamma.ppf(.95, len(g))/len(g)) for g in groups])
    u = np.linalg.solve(b.conj().T@b+variance*np.eye(stats.support), b.conj().T@target)
    def energy(u0):
        r = target-b@u0
        e = np.asarray([np.mean(np.abs(r[g])**2)/variance for g in groups])
        rho = np.where(e<=kappas**2, e, 2*kappas*np.sqrt(e)-kappas**2)
        return e, float(np.vdot(u0, u0).real + sum(len(g)*v for g,v in zip(groups,rho)))
    history = [energy(u)[1]]; last_weights = np.ones(len(groups))
    for _ in range(iterations):
        e, _ = energy(u); last_weights = np.minimum(1., kappas/np.sqrt(np.maximum(e, 1e-30)))
        w = np.empty(len(y))
        for g, v in zip(groups, last_weights): w[g] = v
        u = np.linalg.solve(b.conj().T@(w[:,None]*b)+variance*np.eye(stats.support), b.conj().T@(w*target))
        history.append(energy(u)[1])
    result = masked_output(basis, stats.mean+root@u, pilots.grid)
    count(ledger, 'classical_estimator_calls'); count(ledger, 'full_fits')
    count(ledger, 'linear_solves', iterations+1); count(ledger, 'factorizations')
    return (result, dict(objective=history, weights=last_weights, kappas=kappas, coefficients=stats.mean+root@u)) if details else result
