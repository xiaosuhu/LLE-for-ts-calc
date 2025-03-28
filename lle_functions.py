import numpy as np
import nolds
import matplotlib.pyplot as plt
from scipy.signal import periodogram, correlate
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import cdist

def compute_fnn(series, tau=1, max_dim=10, R_tol=15.0, A_tol=2.0):
    N = len(series)
    fnn_percentages = []

    for m in range(1, max_dim + 1):
        # Reconstruct m-D phase space
        M = N - (m + 1) * tau
        X_m = np.array([series[i : i + m * tau : tau] for i in range(M)])
        X_m1 = np.array([series[i : i + (m + 1) * tau : tau] for i in range(M)])

        false_neighbors = 0
        for i in range(M):
            dists = np.linalg.norm(X_m[i] - X_m, axis=1)
            dists[i] = np.inf  # exclude self
            j = np.argmin(dists)

            R = dists[j]
            delta = abs(X_m1[i][-1] - X_m1[j][-1])  # extra dimension
            A = np.linalg.norm(X_m1[i] - X_m1[j])

            if (delta / R > R_tol) or (A / np.std(series) > A_tol):
                false_neighbors += 1

        fnn_percent = false_neighbors / M
        fnn_percentages.append(fnn_percent)

    return fnn_percentages

# ----------------------------------------
# Phase Space Reconstruction
# ----------------------------------------
def reconstruct_phase_space(x, emb_dim, tau):
    N = len(x) - (emb_dim - 1) * tau
    return np.array([x[i:i + emb_dim * tau:tau] for i in range(N)])

# ----------------------------------------
# Estimate Time Delay (tau) using Mutual Information
# ----------------------------------------
def estimate_tau(series, max_tau=100):
    mi_vals = []
    for tau in range(1, max_tau):
        x = series[:-tau].reshape(-1, 1)
        y = series[tau:]
        mi = mutual_info_regression(x, y)
        mi_vals.append(mi[0])
    tau_opt = np.argmin(mi_vals) + 1  # First minimum
    return tau_opt

# ----------------------------------------
# Estimate Embedding Dimension (FNN method)
# ----------------------------------------
def estimate_emb_dim(series, tau, max_dim=10, threshold=0.05):
    fnn_vals = compute_fnn(series, tau=tau, max_dim=max_dim)
    for i, val in enumerate(fnn_vals):
        if val < threshold:
            return i + 1
    return max_dim

# ----------------------------------------
# Estimate Mean Period using FFT
# ----------------------------------------
def estimate_period(data, fs=1.0):
    f, Pxx = periodogram(data, fs=fs)
    f = f[1:]  # skip DC
    Pxx = Pxx[1:]
    peak_freq = f[np.argmax(Pxx)]
    return 1 / peak_freq  # in seconds

# ----------------------------------------
# Estimate Largest Lyapunov Exponent (Rosenstein)
# ----------------------------------------
def estimate_lyap_r(data, emb_dim, tau, theiler, max_t=50, dt=1.0):
    X = reconstruct_phase_space(data, emb_dim, tau)
    N = len(X)

    # Compute pairwise distances
    D = cdist(X, X)
    for i in range(N):
        D[i, max(0, i - theiler):i + theiler + 1] = np.inf
    neighbors = np.argmin(D, axis=1)

    # Compute divergence over time
    divergence = []
    for k in range(1, max_t):
        d = []
        for i in range(N - k):
            j = neighbors[i]
            if j + k >= N or i + k >= N:
                continue
            dist = np.linalg.norm(X[i + k] - X[j + k])
            if dist > 0:
                d.append(np.log(dist))
        if len(d) > 0:
            divergence.append(np.mean(d))
        else:
            divergence.append(np.nan)

    # Linear fit
    t_vals = np.arange(1, max_t) * dt
    divergence = np.array(divergence)
    mask = ~np.isnan(divergence)
    slope, _ = np.polyfit(t_vals[mask], divergence[mask], deg=1)

    # Plot divergence curve
    """
    plt.plot(t_vals, divergence, label='Avg log divergence')
    plt.xlabel("Time")
    plt.ylabel("ln(distance)")
    plt.title("Rosenstein LLE Estimation")
    plt.grid(True)
    plt.legend()
    plt.show()
    """
    return slope  # LLE

# ----------------------------------------
# MAIN FUNCTION: Automatically Estimate Everything
# ----------------------------------------
def estimate_LLE_auto(data, fs=1.0, max_tau=100, max_dim=10, max_t=50):
    dt = 1.0 / fs
    tau = estimate_tau(data, max_tau)
    print(f"Estimated tau (delay): {tau}")
    
    emb_dim = estimate_emb_dim(data, tau, max_dim=max_dim)
    print(f"Estimated embedding dimension: {emb_dim}")
    
    period = estimate_period(data, fs=fs)
    theiler = int(period / dt)
    print(f"Estimated Theiler window: {theiler} (from period {period:.2f}s)")

    lle = estimate_lyap_r(data, emb_dim, tau, theiler, max_t=max_t, dt=dt)
    print(f"Estimated LLE: {lle:.4f}")
    return lle