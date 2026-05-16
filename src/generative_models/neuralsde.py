import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .helpers import GenerativeBase


class _NeuralSDENet(nn.Module):
    def __init__(self, latent_dim: int, n_tenors: int, hidden: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.drift = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim))
        
        self.diffusion = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.Tanh(), nn.Linear(hidden, latent_dim),
            nn.Softplus())
        
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_tenors))
        
        self.init_encoder = nn.Sequential(
            nn.Linear(n_tenors, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * latent_dim))

    def _ft(self, z, t):
        tt = t.expand(z.size(0), 1)
        return self.drift(torch.cat([z, tt], dim=-1))

    def _gt(self, z, t):
        tt = t.expand(z.size(0), 1)
        return self.diffusion(torch.cat([z, tt], dim=-1))

    def sample_initial(self, x0):
        params     = self.init_encoder(x0)
        mu, logvar = params.chunk(2, dim=-1)
        std        = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std), mu, logvar

    def rollout(self, z0, n_steps: int, dt: float):
        zs = [z0]
        z = z0
        sqrt_dt = dt ** 0.5
        
        for k in range(n_steps):
            t = torch.tensor(k * dt, device=z.device)
            f = self._ft(z, t)
            g = self._gt(z, t)
            z = z + f * dt + g * sqrt_dt * torch.randn_like(z)
            zs.append(z)
        
        Z = torch.stack(zs, dim=1)
        return self.readout(Z)


class NeuralSDEModel(GenerativeBase):
    name = 'Neural Stochastic Differntial Equation'
    display_key = 'neural-sde'

    def __init__(self, seq_len: int, n_tenors: int, latent_dim: int=6, hidden: int=64, device: str=None):
        super().__init__(seq_len, n_tenors, device)
        self.latent_dim = latent_dim
        self.net = _NeuralSDENet(latent_dim, n_tenors, hidden).to(self.device)

    # --------------------------- Training
    def fit(self, matrix, epochs: int=100, batch_size: int=64, lr: float=1e-3, val_fraction: float=0.1, verbose: bool=True, dt: float=1/250):
        X = self._to_array(matrix)
        self._fit_scaler(X)
        
        windows = self._make_windows(self._scale(X))
        n_val = max(1, int(len(windows) * val_fraction))
        train = torch.tensor(windows[:-n_val], device=self.device)
        val = torch.tensor(windows[-n_val:],  device=self.device)
        loader = DataLoader(TensorDataset(train), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        for ep in range(epochs):
            self.net.train()
            losses = []
            for (batch,) in loader:
                x0 = batch[:, 0]
                z0, mu, logvar = self.net.sample_initial(x0)
                y_hat = self.net.rollout(z0, self.seq_len - 1, dt)
                recon = ((y_hat - batch) ** 2).mean()
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                loss = recon + 0.1 * kl
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            
            self.net.eval()
            with torch.no_grad():
                z0v, _, _ = self.net.sample_initial(val[:, 0])
                y_v = self.net.rollout(z0v, self.seq_len - 1, dt)
                val_loss = ((y_v - val) ** 2).mean().item()
            
            self.history['train_loss'].append(float(np.mean(losses)))
            self.history['val_loss'].append(val_loss)
            
            if verbose and (ep % max(1, epochs // 20) == 0 or ep == epochs - 1):
                print(f'NeuralSDE: epoch {ep+1:3d}/{epochs} | train={np.mean(losses):.6f} | val={val_loss:.6f}')

        self._fitted = True
        return self

    # --------------------------- Generation
    def generate(self, n_samples: int, horizon: int=None, dt: float=1/250, init_yields: np.ndarray=None) -> np.ndarray:
        self._check_fitted()
        horizon = horizon or self.seq_len
        self.net.eval()
        
        with torch.no_grad():
            if init_yields is None:
                x0 = torch.zeros(n_samples, self.n_tenors, device=self.device)
            else:
                x0_scaled = self._scale(np.atleast_2d(init_yields)).astype(np.float32)
                if x0_scaled.shape[0] == 1:
                    x0_scaled = np.repeat(x0_scaled, n_samples, axis=0)
                x0 = torch.tensor(x0_scaled, device=self.device)
            z0, _, _ = self.net.sample_initial(x0)
            y_hat = self.net.rollout(z0, horizon - 1, dt).cpu().numpy()
        return self._unscale(y_hat)