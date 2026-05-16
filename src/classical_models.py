import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from .display_utils import print_params_box


TENOR_YEARS = {'1W': 7/365, '2W': 14/365, '1M': 30/365, '2M': 60/365, '3M': 90/365, '6M': 180/365, '1Y': 1.0, '2Y': 2.0}

_DISPLAY_KEYS = {
    'vasicek': 'vasicek',
    'hull-white': 'hull-white',
    'dns': 'dynamic-nelson-siegel'}

AVAILABLE_MODELS = list(_DISPLAY_KEYS.keys())


class ClassicModel:
    def __init__(self, model: str, lam: float = 0.7308):
        model = model.lower().strip()
        if model not in AVAILABLE_MODELS:
            print(f"Unknown model '{model}'.\nChoose from: {AVAILABLE_MODELS}")
        
        self.model = model
        self.lam = lam
        self.params = {}
        self._n_obs = 0
        self.is_fitted = False

    def calibrate(self, data, dt=1/250, taus=None):
        if self.model == 'vasicek':
            self.params = self._calibrate_vasicek(data, dt)
        elif self.model == 'hull-white':
            self.params = self._calibrate_hull_white(data, dt)
        elif self.model == 'hull-white-2f':
            self.params = self._calibrate_hull_white_2f(data, dt)
        elif self.model == 'nelson-siegel':
            if taus is None:
                print("'nelson-siegel' requires taus argument.")
            self.params = self._calibrate_ns(taus, data)
        elif self.model == 'dns':
            if taus is None:
                print("'dns' requires taus argument.")
            self.params = self._calibrate_dns(data, taus)

        self.is_fitted = True
        return self

    def simulate(self, T, dt=1/250, n_paths=100, seed=42, plot=True, n_show=50):
        self._check_fitted()

        if self.model == 'vasicek':
            arr = self._simulate_vasicek(T, dt, n_paths, seed)
        elif self.model == 'hull-white':
            arr = self._simulate_hull_white(T, dt, n_paths, seed)
        elif self.model == 'dns':
            arr = self._simulate_dns_level(T, dt, n_paths, seed)

        paths = pd.DataFrame(arr, columns=[f'path_{i+1}' for i in range(n_paths)])

        if plot:
            self._plot_paths(paths, T=T, r0=self.params['r0'], title=self.model.upper(), n_show=n_show)

        return paths

    def simulate_curves(self, taus, T, dt=1/250, n_paths=1000, seed=42):
        self._check_fitted()
        
        if self.model != 'dns':
            print("simulate_curves() is only available for model='dns'.")
        return self._simulate_dns_curves(taus, T, dt, n_paths, seed)

    def predict(self, taus):
        self._check_fitted()
        
        if self.model != 'nelson-siegel':
            print("predict() is only available for model='nelson-siegel'.")
        p = self.params
        
        return self._ns_curve(np.asarray(taus, dtype=float), p['beta0'], p['beta1'], p['beta2'], p['lambda'])

    def display_params(self):
        self._check_fitted()
        shown = {}

        for k, v in self.params.items():
            if isinstance(v, np.ndarray) and v.size > 6:
                shown[f'{k} (summary)'] = (f'len={v.size},  mean={v.mean():.4f},  std={v.std():.4f}')
            else:
                shown[k] = v
        
        print_params_box(shown, title=self.model.upper(), model_key=_DISPLAY_KEYS[self.model], subtitle=f'calibrated on {self._n_obs} observations')

    # -------------------- Calibaration --------------------
    def _calibrate_vasicek(self, r, dt):
        r = self._clean(r)
        
        slope, intercept, *_ = stats.linregress(r[:-1], r[1:])
        slope     = float(np.clip(slope, 1e-8, 0.9999))
        residuals = r[1:] - (intercept + slope * r[:-1])
        kappa     = -np.log(slope) / dt
        theta     = intercept / (1 - slope)
        sigma     = residuals.std(ddof=2) * np.sqrt(2 * kappa / (1 - slope**2))
        self._n_obs = len(r)
        return {'kappa': kappa, 'theta': theta, 'sigma': sigma, 'r0': float(r[-1])}

    def _calibrate_hull_white(self, r, dt):
        r = self._clean(r)
        
        slope, intercept, *_ = stats.linregress(r[:-1], r[1:])
        slope = float(np.clip(slope, 1e-8, 0.9999))
        residuals = r[1:] - (intercept + slope * r[:-1])
        a = -np.log(slope) / dt
        sigma = residuals.std(ddof=2) * np.sqrt(2 * a / (1 - slope**2))
        theta_t = np.gradient(r, dt) + a * r
        self._n_obs = len(r)
        return {'a': a, 'sigma': sigma, 'theta_t': theta_t, 'r0': float(r[-1])}

    def _calibrate_ns(self, taus, yields):
        taus = np.asarray(taus, dtype=float)
        yields = np.asarray(yields, dtype=float)
        mask = ~np.isnan(yields)
        taus, yields = taus[mask], yields[mask]
        p0 = [yields[-1], yields[0] - yields[-1], 0.0, 1.0]
        bounds = ([-100]*3 + [0.01], [100]*3 + [10.0])
        popt, _ = curve_fit(self._ns_curve, taus, yields, p0=p0, bounds=bounds)
        self._n_obs = int(yields.size)

        return {
            'beta0': float(popt[0]),
            'beta1': float(popt[1]),
            'beta2': float(popt[2]),
            'lambda': float(popt[3]),
        }

    def _calibrate_dns(self, yield_matrix, taus):
        Y = yield_matrix.dropna().values
        taus = np.asarray(taus, dtype=float)
        ts = np.where(taus == 0, 1e-6, taus)
        
        # loading funxtions
        g1 = (1 - np.exp(-self.lam * ts)) / (self.lam * ts)
        g2 = g1 - np.exp(-self.lam * ts)
        
        X = np.column_stack([np.ones_like(ts), g1, g2])
        betas = (np.linalg.pinv(X.T @ X) @ X.T @ Y.T).T  # (T, 3)
        B0, B1 = betas[:-1], betas[1:]
        Z = np.column_stack([np.ones(len(B0)), B0])
        coef, *_ = np.linalg.lstsq(Z, B1, rcond=None)
        resid = B1 - Z @ coef
        
        self._n_obs = len(Y)
        return {
            'lambda': float(self.lam),
            'mu': coef[0],
            'A': coef[1:].T,
            'Sigma': np.cov(resid.T, ddof=3),
            'beta_last': betas[-1],
            'r0': float(Y[-1, 0]),
        }

    # -------------------- Simulation --------------------
    def _simulate_vasicek(self, T, dt, n_paths, seed):
        rng = np.random.default_rng(seed)
        N = int(round(T / dt))
        kappa, theta, sigma, r0 = (self.params[k] for k in ('kappa', 'theta', 'sigma', 'r0'))
        paths = np.empty((N, n_paths))
        paths[0] = r0
        for t in range(1, N):
            paths[t] = (paths[t-1] + kappa * (theta - paths[t-1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths))
        return paths

    def _simulate_hull_white(self, T, dt, n_paths, seed):
        rng = np.random.default_rng(seed)
        N   = int(round(T / dt))
        a, sigma, theta_t, r0 = (self.params[k] for k in ('a', 'sigma', 'theta_t', 'r0'))
        
        if theta_t.size >= N:
            theta_fwd = theta_t[-N:]
        else:
            theta_fwd = np.concatenate([theta_t, np.full(N - theta_t.size, theta_t[-1])])
        paths    = np.empty((N, n_paths))
        paths[0] = r0
        
        for t in range(1, N):
            paths[t] = (paths[t-1] + (theta_fwd[t-1] - a * paths[t-1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths))
        return paths

    def _simulate_dns_level(self, T, dt, n_paths, seed):
        rng = np.random.default_rng(seed)
        N = int(round(T / dt))
        mu = self.params['mu']
        A = self.params['A']
        L = np.linalg.cholesky(self.params['Sigma'] + 1e-10 * np.eye(3))
        b0 = self.params['beta_last']

        level = np.empty((N, n_paths))
        for sim in range(n_paths):
            beta = b0.copy()
            for t in range(N):
                level[t, sim] = beta[0]
                beta = mu + A @ beta + L @ rng.standard_normal(3)
        return level

    def _simulate_dns_curves(self, taus, T, dt, n_paths, seed):
        rng = np.random.default_rng(seed)
        N = int(round(T / dt))
        mu = self.params['mu']
        A = self.params['A']
        L = np.linalg.cholesky(self.params['Sigma'] + 1e-10 * np.eye(3))
        b0 = self.params['beta_last']
        taus = np.asarray(taus, dtype=float)
        ts = np.where(taus == 0, 1e-6, taus)
        g1 = (1 - np.exp(-self.lam * ts)) / (self.lam * ts)
        g2 = g1 - np.exp(-self.lam * ts)
        X = np.column_stack([np.ones_like(ts), g1, g2])

        curves = np.empty((N, n_paths, len(taus)))
        for sim in range(n_paths):
            beta = b0.copy()
            for t in range(N):
                curves[t, sim] = X @ beta
                beta = mu + A @ beta + L @ rng.standard_normal(3)
        return curves
    
    # -------------------- Helpers --------------------
    @staticmethod
    def _clean(r):
        r = np.asarray(r, dtype=float)
        return r[~np.isnan(r)]

    @staticmethod
    def _ns_curve(tau, beta0, beta1, beta2, lam):
        tau = np.where(tau == 0, 1e-6, tau)
        g1  = (1 - np.exp(-lam * tau)) / (lam * tau)
        g2  = g1 - np.exp(-lam * tau)
        return beta0 + beta1 * g1 + beta2 * g2

    def _check_fitted(self):
        if not self.is_fitted:
            print(f"Model '{self.model}' is not calibrated yet. Call .calibrate() first.")
    
    def _plot_paths(self, paths, T, r0, title, n_show=50, figsize=(12, 5)):
        time_axis = np.linspace(0, T, len(paths))

        fig, ax = plt.subplots(figsize=figsize)

        for col in paths.columns[:n_show]:
            ax.plot(time_axis, paths[col].values, linewidth=0.5, alpha=0.3, color='blue')

        ax.plot(time_axis, paths.mean(axis=1).values, linewidth=2, color='red', label='Mean')
        ax.fill_between(time_axis, paths.quantile(0.05, axis=1).values, paths.quantile(0.95, axis=1).values, alpha=0.12, color='blue', label='5th–95th pct')
        ax.axhline(r0, color='grey', linewidth=1, linestyle='--', label=f'r0 = {r0:.2f}%')
        ax.set_xlabel('Time in years')
        ax.set_ylabel('Rate in % per annum')
        ax.set_title(f'{title}: {len(paths.columns)} paths for T={T:.3f} year(-s)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.show()