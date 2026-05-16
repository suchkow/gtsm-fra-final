import numpy as np
import pandas as pd
import torch
from ..display_utils import print_params_box


class GenerativeBase:
    name = 'base'
    display_key = 'base'

    def __init__(self, seq_len: int, n_tenors: int, device: str=None):
        self.seq_len  = int(seq_len)
        self.n_tenors = int(n_tenors)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.net = None
        self.history = {'train_loss': [], 'val_loss': []}
        self._scale_mean = None
        self._scale_std = None
        self._fitted = False

    def _fit_scaler(self, data: np.ndarray) -> None:
        self._scale_mean = data.mean(axis=0)
        self._scale_std  = data.std(axis=0) + 1e-8

    def _scale(self, data: np.ndarray) -> np.ndarray:
        return (data - self._scale_mean) / self._scale_std

    def _unscale(self, data: np.ndarray) -> np.ndarray:
        return data * self._scale_std + self._scale_mean

    def _make_windows(self, matrix: np.ndarray) -> np.ndarray:
        T = matrix.shape[0]
        n_windows = T - self.seq_len + 1
        
        if n_windows <= 0:
            raise ValueError(f'History too short ({T}) for seq_len={self.seq_len}.')
        out = np.empty((n_windows, self.seq_len, self.n_tenors), dtype=np.float32)
        for i in range(n_windows):
            out[i] = matrix[i : i + self.seq_len]
        return out

    @staticmethod
    def _to_array(matrix) -> np.ndarray:
        return matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)

    def _check_fitted(self) -> None:
        if not self._fitted:
            print(f"Model '{self.name}' is not fitted yet. Call .fit() first.")

    def display_training_info(self) -> None:
        self._check_fitted()
        info = {'architecture': self.name,
                'device': str(self.device),
                'seq_len': self.seq_len,
                'n_tenors': self.n_tenors,
                'n_parameters': sum(p.numel() for p in self.net.parameters()),
                'epochs_run': len(self.history['train_loss']),
                'final train loss': self.history['train_loss'][-1],
                'best val loss': (min(self.history['val_loss']) if self.history['val_loss'] else float('nan'))}
        
        print_params_box(info, title=f'{self.name}. Training summary', model_key=self.display_key)

    def save(self, path: str) -> None:
        torch.save({'state_dict': self.net.state_dict(),
                    'scale_mean': self._scale_mean,
                    'scale_std': self._scale_std,
                    'history': self.history,
                    'seq_len': self.seq_len,
                    'n_tenors': self.n_tenors,
                    'name': self.name}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['state_dict'])
        self._scale_mean = ckpt['scale_mean']
        self._scale_std = ckpt['scale_std']
        self.history = ckpt['history']
        self._fitted = True
        return self