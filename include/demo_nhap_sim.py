import numpy as np
import scipy.stats as si
from scipy import optimize
import matplotlib.pyplot as plt

# BS call price and implied vol functions (from provided code)
def calc_impl_volatility(S, K, r, q, T, P):
    """Implied volatility via root-finding on (market price - BS price)."""
    P_adj = P

    def price_comp(sigma):
        return P_adj - call_price(S, K, r, q, sigma, T)

    v = None
    t = 0
    s = -1
    # Add tiny noise to price if brentq fails to bracket a root
    while v is None and t < 20:
        P_adj = P + t * s * 0.0001
        try:
            v = optimize.brentq(price_comp, 0.001, 0.5, xtol=10**(-12), rtol=10**(-12), maxiter=10000)
        except Exception:
            v = None
            if s > 0:
                t += 1
            s *= -1

    return v

def call_price(S, K, r, q, v, T):
    if T <= 0.0:
        return max(S - K, 0.0)
    d1, d2 = _d(S, K, r, q, v, T)
    N1, N2 = _N(d1, d2)
    return S * np.exp(-q * T) * N1 - K * np.exp(-r * T) * N2

def _d(S, K, r, q, v, T):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return d1, d2

def _N(d1, d2):
    return si.norm.cdf(d1), si.norm.cdf(d2)

# Parameters (fixed S, T, r, q, v; vary K)
S = 100.0
r = 0.05
q = 0.0
v = 0.2  # Fixed volatility
T = 1.0  # 1 year maturity

# Range of strikes K (e.g., from 80% to 120% of S)
Ks = np.linspace(90, 115, 2)

# Compute implied vols using call_price as "market" price
ivs = []
for K in Ks:
    # Get "market" price from Black-Scholes call_price
    P_bs = call_price(S, K, r, q, v, T)
    
    # Compute BS implied vol from that price (should recover v)
    iv = calc_impl_volatility(S, K, r, q, T, P_bs)
    ivs.append(iv)
print(ivs)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(Ks, ivs, marker='o')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility (Black-Scholes Model Prices)')
plt.grid(True)
plt.show()