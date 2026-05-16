import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .helpers import GenerativeBase


class _SeqVAENet(nn.Module):
    def __init__(self, n_tenors: int, hidden: int=64, latent: int=8):
        super().__init__()
        
        self.enc_gru = nn.GRU(n_tenors, hidden, batch_first=True)
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_logvar = nn.Linear(hidden, latent)
        self.dec_init = nn.Linear(latent, hidden)
        self.dec_gru = nn.GRU(n_tenors, hidden, batch_first=True)
        self.dec_out = nn.Linear(hidden, n_tenors)
        self.latent = latent

    def encode(self, x):
        _, h = self.enc_gru(x)
        h = h.squeeze(0)
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z, seq_len, x_teacher=None):
        h0 = self.dec_init(z).unsqueeze(0)
        
        if x_teacher is not None:
            shifted = torch.cat([torch.zeros_like(x_teacher[:, :1]), x_teacher[:, :-1]], dim=1)
            out, _ = self.dec_gru(shifted, h0)
            return self.dec_out(out)
        
        batch = z.size(0)
        x_prev = torch.zeros(batch, 1, self.dec_out.out_features, device=z.device)
        h = h0
        outs = []
        
        for _ in range(seq_len):
            o, h = self.dec_gru(x_prev, h)
            y = self.dec_out(o)
            outs.append(y)
            x_prev = y
        
        return torch.cat(outs, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, x.size(1), x_teacher=x)
        return recon, mu, logvar


class VAEModel(GenerativeBase):
    name = 'Varitaional Autoencoder'
    display_key = 'vae'

    def __init__(self, seq_len: int, n_tenors: int, hidden: int=64, latent: int=8, device: str=None,):
        super().__init__(seq_len, n_tenors, device)
        self.latent_dim = latent
        self.net = _SeqVAENet(n_tenors, hidden, latent).to(self.device)

    def fit(self, matrix, epochs: int=100, batch_size: int=64, lr: float=0.001, val_fraction: float=0.1, verbose: bool=True, kl_weight: float=1.0):
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
            for (batch, ) in loader:
                recon, mu, logvar = self.net(batch) # using forward()
                recon_loss = ((recon - batch) ** 2).mean()
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                loss = recon_loss + kl_weight * kl
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                
            self.net.eval()

            with torch.no_grad():
                recon, mu, logvar = self.net(val)
                val_loss = (((recon - val) ** 2).mean() - 0.5 * kl_weight * (1 + logvar - mu.pow(2) - logvar.exp()).mean()).item()
            
            self.history['train_loss'].append(float(np.mean(losses)))
            self.history['val_loss'].append(val_loss)
            
            if verbose and (ep % max(1, epochs // 20) == 0 or ep == epochs - 1):
                print(f'VAE: epoch {ep+1:3d}/{epochs} | train={np.mean(losses):.6f} | val={val_loss:.6f}')

        self._fitted = True
        return self

    def generate(self, n_samples: int, horizon: int=None) -> np.ndarray:
        self._check_fitted()
        horizon = horizon or self.seq_len
        self.net.eval()
        
        with torch.no_grad():
            z   = torch.randn(n_samples, self.latent_dim, device=self.device)
            out = self.net.decode(z, horizon).cpu().numpy()
        
        return self._unscale(out)