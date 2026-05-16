from dataclasses import dataclass
from typing import Any, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .cbr_parser import CBRParser
from .display_utils import styled_stats


TENOR_YEARS = {'1W': 7/365, '2W': 14/365, '1M': 30/365, '2M': 60/365, '3M': 90/365, '6M': 180/365, '1Y': 1.0, '2Y': 2.0}

@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __iter__(self):
        return iter((self.train, self.val, self.test))


class DataHandler:
    # -------------------- Init --------------------
    def __init__(self, df: pd.DataFrame, tenors = None):
        self.df_raw = df.copy()
        self.df = None
        self.tenors = (list(tenors) if tenors is not None else [c for c in df.columns if c != 'date'])

    @classmethod
    def from_excel(cls, path: str, date_col: str = 'date'):
        df = pd.read_excel(path)
        df = df.rename(columns={date_col: 'date'})
        return cls(df)

    @classmethod
    def from_cbr(cls, start: str, end: str, parser: CBRParser):
        parser = parser or CBRParser()
        df = parser.roisfix(start, end)
        return cls(df)

    # -------------------- Cleaning --------------------
    def clean(self, treat_zero_as_missing: bool = True, sort_ascending: bool = True):
        df = self.df_raw.copy()

        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['date'])

        if treat_zero_as_missing:
            for t in self.tenors:
                df.loc[df[t] == 0, t] = np.nan

        df = df.sort_values('date', ascending=sort_ascending).reset_index(drop=True)
        self.df = df
        return self

    # -------------------- Helpers --------------------
    def _require_clean(self) -> pd.DataFrame:
        if self.df is None:
            print('Call .clean() first.')
        return self.df

    def yield_matrix(self, dropna: bool = True) -> pd.DataFrame:
        df = self._require_clean().set_index('date')[self.tenors]
        return df.dropna() if dropna else df

    def differences(self, scale: int = 100, dropna: bool = True) -> pd.DataFrame:
        out = self.yield_matrix(dropna=dropna).diff()*scale
        return out.dropna() if dropna else out

    def log_returns(self) -> pd.DataFrame:
        m = self.yield_matrix()
        return np.log(m).diff().dropna()


    # -------------------- Plots & Statistics --------------------
    def descriptive_statistics(self, caption: str, style: bool = True):
        caption = caption or 'Descriptive Statistics'
        m = self.yield_matrix(dropna=False)

        stats = pd.DataFrame({
                'count': m.count(),
                'mean': m.mean(),
                'std': m.std(),
                'min': m.min(),
                '25%': m.quantile(0.25),
                '50%': m.quantile(0.50),
                '75%': m.quantile(0.75),
                'max': m.max(),
                'skew': m.skew(),
                'kurtosis': m.kurtosis()}
                ).T

        if style:
            return styled_stats(stats, caption=caption, fmt='{:.4f}')
        return stats

    def diff_statistics(self, caption: str, style: bool = True) -> Any:
        caption = caption or 'Descriptive Statistics'
        d = self.differences()
        
        stats = pd.DataFrame({
                'count': d.count(),
                'mean': d.mean(),
                'std': d.std(),
                'min': d.min(),
                'max': d.max(),
                'skew': d.skew(),
                'kurtosis': d.kurtosis()}
                ).T
        
        if style:
            return styled_stats(stats, caption=caption, fmt='{:.4f}')
        return stats

    def plot_time_series(self) -> None:
        
        df = self._require_clean()
        ax = plt.subplots(figsize=(12, 5))[1]
        
        for t in self.tenors:
            series = df.set_index('date')[t].dropna()
            if len(series):
                ax.plot(series.index, series.values, label=t, linewidth=1.2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Yield in % per annum')
        ax.set_title('Yield dynamics by tenor')
        ax.legend(loc='best', ncol=4, fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_yield_curves(self, dates, n_recent=5) -> None:
        df = self._require_clean().set_index('date')
        
        if dates is None:
            chosen = df.dropna(how='all').index[-n_recent:]
        else:
            chosen = pd.to_datetime(list(dates))

        taus = [TENOR_YEARS[t] for t in self.tenors]
        fig, ax = plt.subplots(figsize=(9, 5))
        for d in chosen:
            if d not in df.index:
                continue
            row = df.loc[d, self.tenors]
            mask = row.notna()
            ax.plot(np.array(taus)[mask], row[mask].values, marker='o', linewidth=1.6, label=pd.Timestamp(d).strftime('%Y-%m-%d'))
        
        ax.set_xlabel('Maturity in years')
        ax.set_ylabel('Yield in % per annum')
        ax.set_title('Yield curve')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_distributions(self, which: str = 'levels', bins: int = 60) -> None:
        if which == 'levels':
            mat = self.yield_matrix()
            xlabel = 'Yield (%)'
        elif which in ('diff', 'differences'):
            mat = self.differences()
            xlabel = 'Daily change (pp)'
        else:
            print(f'Unknown which: {which}')

        n = len(self.tenors)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
        axes = np.atleast_2d(axes).ravel()

        for i, tenor in enumerate(self.tenors):
            ax = axes[i]
            data = mat[tenor].dropna().values
            if len(data) == 0:
                ax.set_visible(False)
                continue
            ax.hist(data, bins=bins, density=True, alpha=0.6, edgecolor='white')
            mu, sigma = data.mean(), data.std()
            xs = np.linspace(data.min(), data.max(), 200)
            ax.plot(xs, _gaussian_pdf(xs, mu, sigma), color='crimson', linewidth=1.2, label=f'N({mu:.2f}, {sigma:.2f})')
            ax.set_title(tenor)
            ax.set_xlabel(xlabel)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

        for j in range(len(self.tenors), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f'Empirical distribution vs. Gaussian fit. [{which}]', y=1.02)
        plt.tight_layout()

    def plot_correlation_matrix(self, which: str = 'differences') -> None:
        data = (self.differences() if which == 'differences' else self.yield_matrix())
        corr = data.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
            
        ax.set_title(f'Correlation matrix. [{which}]')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

    def plot_pca_decomposition(self, n_components: int = 3) -> None:
        diffs = self.differences()
        pca = PCA(n_components=n_components)
        pca.fit(diffs.values)

        taus = [TENOR_YEARS[t] for t in self.tenors]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        labels = ['Level', 'Slope', 'Curvature'] + [f'Component {i+1}' for i in range(3, n_components)]
        for i in range(n_components):
            axes[0].plot(taus, pca.components_[i], marker='o', label=f'{labels[i]} ({pca.explained_variance_ratio_[i]:.1%})')
        
        axes[0].axhline(0, color='black', linewidth=0.5)
        axes[0].set_xlabel('Maturity in years')
        axes[0].set_ylabel('Loading')
        axes[0].set_title('PCA Loadings')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(range(1, n_components + 1), pca.explained_variance_ratio_, color='blue')
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('Explained Variance Share (Eigenvalue)')
        axes[1].set_title('PCA Scree')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

    # -------------------- Splits --------------------
    def train_val_test_split(self, train_end: str, val_end: str, plot: bool = False):
        df = self.yield_matrix(dropna=False)
        
        t_end = pd.Timestamp(train_end)
        v_end = pd.Timestamp(val_end)
        
        train = df.loc[df.index <= t_end]
        val   = df.loc[(df.index > t_end) & (df.index <= v_end)]
        test  = df.loc[df.index > v_end]
        split = DataSplit(train=train, val=val, test=test)
 
        if plot:
            total = len(train) + len(val) + len(test)
            
            ref_tenor = next((t for t in self.tenors if df[t].notna().sum() > 100), self.tenors[0])
            series = df[ref_tenor]
            
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.axvspan(train.index.min(), t_end, alpha=0.12, color='blue', label=f'Train: {len(train):,} obs ({len(train)/total:.0%})')
            ax.axvspan(t_end, v_end, alpha=0.12, color='orange', label=f'Validation: {len(val):,} obs ({len(val)/total:.0%})')
            ax.axvspan(v_end, test.index.max(), alpha=0.12, color='green', label=f'Test: {len(test):,} obs ({len(test)/total:.0%})')
            
            ax.plot(series.index, series.values, color='black', linewidth=0.9, alpha=0.85, label=ref_tenor)
            ymax = series.dropna().max()
            
            for xval, label in [(t_end, train_end), (v_end, val_end)]:
                ax.axvline(xval, color='grey', linewidth=1.2, linestyle='--')
                ax.text(xval, ymax, f'{label}', fontsize=8, color='grey', va='top')
            
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Yield in % per annum. Tenor: {ref_tenor}')
            ax.set_title('Train, Validation and Test Splits')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.show()
 
        return split

# -------------------- Module level functions --------------------
def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float):
    if sigma <= 0:
        return np.zeros_like(x)
    return ( 1.0 / (sigma * np.sqrt(2 * np.pi)) ) * np.exp( -0.5 * ((x - mu) / sigma) ** 2 )