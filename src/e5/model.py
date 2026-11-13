"""Small pilot-only residual; no generator or evaluation imports."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from src.estimators.features import pilot_features
from src.estimators.practical import lmmse_like
from src.estimators.noise import estimate_variance
from src.ofdm.information import require_pilots
from src.utils.cost import count

class ResidualCNN(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(7,width,3,padding=1),nn.ReLU(),
                                  nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
                                  nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
                                  nn.Conv2d(width,2,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=7:raise ValueError('Expected [B,7,N,L]')
        return self.layers(x)*x[:,5:6]

@dataclass
class CandidateOperator:
    statistics: object
    model: ResidualCNN
    device: str='cpu'
    variance_floor: float=1e-10
    solve_floor: float=1e-8
    scale_floor: float=1e-12

    def base(self,pilots,ledger=None):
        require_pilots(pilots)
        return lmmse_like(pilots,self.statistics,self.variance_floor,self.solve_floor,ledger)

    def neural(self,pilots,base,variance=None,ledger=None):
        require_pilots(pilots)
        variance=estimate_variance(pilots,self.variance_floor,ledger) if variance is None else variance
        features,scale=pilot_features(pilots,base,variance,self.scale_floor)
        with torch.inference_mode():
            result=self.model(torch.as_tensor(features,dtype=torch.float32,device=self.device))
            delta=result[0].detach().cpu().numpy()
        count(ledger,'network_passes')
        count(ledger,'network_partial_passes' if not np.array_equal(pilots.fit_mask,pilots.grid.pilot_mask) else 'network_full_passes')
        correction=scale*(delta[0].astype(np.float64)+1j*delta[1].astype(np.float64))
        candidate=base+correction
        if not np.isfinite(candidate).all():raise ArithmeticError('Nonfinite neural candidate; abort')
        return candidate

    def pair(self,pilots,ledger=None):
        base=self.base(pilots,ledger)
        variance=estimate_variance(pilots,self.variance_floor,ledger)
        return base,self.neural(pilots,base,variance,ledger),variance
