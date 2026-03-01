# option_functions.py
# Minimal fix for SciPy ≥1.13: use numpy for exp/log, keep scipy.optimize.

import numpy as np
from scipy import optimize
import QuantLib as ql
import scipy.stats as si
from scipy.stats import norm

from math import log, sqrt, pi
from scipy.integrate import quad

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
            v = optimize.brentq(price_comp, 0.001, 100.0, maxiter=1000)
        except Exception:
            v = None
            if s > 0:
                t += 1
            s *= -1

    return v


def calc_sabr_impl_volatility(
    F: float,           # forward price F = S₀ * exp((r-q)T)
    K: float,           # strike price
    T: float,           # time to maturity in years
    alpha: float,       # SABR α parameter (initial vol level)
    rho: float,         # correlation between forward and volatility
    nu: float           # vol-of-vol parameter
) -> float:
    """
    Compute the implied Black volatility σ_B under SABR model (β=1 case)
    using Hagan et al. (2002) approximation.

    Parameters:
    -----------
    F      : float          Forward price
    K      : float          Strike price
    T      : float          Time to maturity (years)
    alpha  : float          SABR α (initial volatility level)
    rho    : float          Correlation parameter (-1 < rho < 1)
    nu     : float          Volatility of volatility

    Returns:
    --------
    float : Implied Black volatility σ_B
    """
    if T <= 0:
        return alpha  # degenerate case

    # ATM case / very small difference
    if abs(F - K) < 1e-10 or T < 1e-12:
        # Quick ATM approximation
        correction = (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho**2) * nu**2 / 24.0) * T
        return alpha * (1.0 + correction)

    # -----------------------------
    # Standard Hagan formula
    # -----------------------------
    z = nu / alpha * np.log(F / K)

    # x(z) function - the tricky part
    # We use safe evaluation to avoid numerical issues near z=0
    if abs(z) < 1e-6:
        # Taylor expansion around z=0: x(z) ≈ z * (1 + (rho*z)/3 + ...)
        x_z = z * (1.0 + rho * z / 3.0 + (rho**2 - 1.0) * z**2 / 12.0)
    else:
        numerator = np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho
        x_z = np.log(numerator / (1.0 - rho))

    # Main term: z / x(z)
    if abs(x_z) < 1e-10:
        ratio = 1.0  # limit when z→0
    else:
        ratio = z / x_z

    # Time-dependent correction term
    correction = (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho**2) * nu**2 / 24.0) * T

    # Final implied volatility
    sigma_B = alpha * ratio * (1.0 + correction)

    return sigma_B





def _d(S, K, r, q, v, T):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return d1, d2


def _N(d1, d2):
    return si.norm.cdf(d1), si.norm.cdf(d2)

def call_price(S, K, r, q, v, T):
    if T <= 0.0:
        return max(S - K, 0.0)
    d1, d2 = _d(S, K, r, q, v, T)
    N1, N2 = _N(d1, d2)
    return S * np.exp(-q * T) * N1 - K * np.exp(-r * T) * N2


def put_price(S, K, r, q, v, T):
    d1, d2 = _d(S, K, r, q, v, T)
    N1, N2 = _N(d1, d2)
    # Use np.exp instead of scipy.exp
    return -S * np.exp(-q * T) * (1.0 - N1) + K * np.exp(-r * T) * (1.0 - N2)



def heston_price(S, K, r, q, theta, kappa, sigma, rho, v0, exp_date, cur_date):
    ql.Settings.instance().evaluationDate = ql.DateParser.parseFormatted(cur_date, '%Y-%m-%d')
    exp_date = ql.DateParser.parseFormatted(exp_date, '%Y-%m-%d')

    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), r, ql.Actual360()))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), q, ql.Actual360()))

    heston_process = ql.HestonProcess(r_handle, q_handle, spot, v0, kappa, theta, sigma, rho)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(exp_date)
    european_option = ql.VanillaOption(payoff, exercise)

    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 0.01, 1000)
    european_option.setPricingEngine(engine)
    return european_option.NPV()




# ---------- RS-GBM ----------
def _rs_mu(S0, q0, q1, r, sig0, sig1, T, t):
    """Mean of log S(T) given occupation time t in state 0."""
    return (
        log(S0)
        + (q1 - q0 - 0.5 * (sig0**2 - sig1**2)) * t
        + (r  - q1 - 0.5 * sig1**2) * T
    )

def _rs_var(sig0, sig1, T, t):
    """Variance of log S(T) given occupation time t in state 0."""
    return (sig0**2 - sig1**2) * t + (sig1**2) * T

def _rs_integrand0(y, S0, q0, q1, r, sig0, sig1, T, K, l0=1.0, l1=1.0):
    """
    Integrand for the starting regime = 0 case.
    l0,l1 are accepted for API-compatibility; only l0 is (optionally) used.
    """
    # Two mixing points (Appendix D style)
    m0T = _rs_mu(S0, q0, q1, r, sig0, sig1, T, T/3.0)
    v0T = _rs_var(sig0, sig1, T, T/3.0)
    mT  = _rs_mu(S0, q0, q1, r, sig0, sig1, T, T)
    vT  = _rs_var(sig0, sig1, T, T)

    z = y + K
    if z <= 0.0:
        return 0.0

    # lognormal pdfs at z
    a = (1.0 / sqrt(2.0 * pi * v0T)) * np.exp(-(np.log(z) - m0T)**2 / (2.0 * v0T))
    b = (1.0 / sqrt(2.0 * pi * vT )) * np.exp(-(np.log(z) - mT )**2 / (2.0 * vT ))

    # If you want to force l0=1.0, pass nothing; if caller supplies l0, it’s used here
    mix = a * (1.0 - np.exp(-l0 * T)) + b * np.exp(-l0 * T)
    return (y / z) * mix

def rs_gbm_call_start0(S0, K, r, q0, q1, sig0, sig1, T, l0=1.0, l1=1.0):
    """
    European call price under RS-GBM when the initial regime is 0.
    l0,l1 are accepted to avoid TypeError if the caller passes them; l1 unused.
    """
    if T <= 0.0:
        return float(max(S0 - K, 0.0))

    val, _ = quad(_rs_integrand0, 0.0, np.inf,
                  args=(S0, q0, q1, r, sig0, sig1, T, K, l0, l1))
    return float(np.exp(-r * T) * val)


# ---------- MJD ----------


def mjd_price(S, K, r, q, sigma, T, lam, m, v, call=True, n_terms=80):
    """
    European option price under Merton Jump-Diffusion model aligned with the provided setting.
 https://quant.stackexchange.com/questions/37565/price-of-european-calls-in-mertons-model, page 21
    Parameters:
    - S: spot price
    - K: strike
    - r: risk-free rate
    - q: dividend yield
    - sigma: diffusion volatility (σ)
    - T: time to maturity (τ)
    - lam: jump intensity (λ)
    - mu: mean of log-jump size (μ)
    - delta: std dev of log-jump size (δ)
    - call: True for call, False for put
    - n_terms: number of terms in series truncation

    The formula conditions on n ~ Poisson(λ T), with adjusted σ_n and r_n.
    """
    if T <= 0.0:
        return call_price(S, K, r, q, sigma, T) if call else put_price(S, K, r, q, sigma, T)

    # Compensator k = E[Y - 1] = exp(μ + δ²/2) - 1
    k = np.exp(m + 0.5 * v**2) - 1.0

    # Poisson mean = λ (k+1)
    pois_mean = lam * (k+1) * T 

    price = 0.0
    # Initial weight for n=0
    weight = np.exp(-pois_mean)
    fact = 1.0  # 0! = 1

    for n in range(n_terms + 1):
        # Adjusted volatility σ_n = sqrt(σ² + n δ² / T)
        sigma_n = np.sqrt(sigma**2 + (n * v**2) / T)

        # Adjusted rate r_n = r - λ k + (n / T) (μ + δ² / 2)
        r_n = r - lam * k + (n / T) * (m + 0.5 * v**2)

        # Black-Scholes price with adjusted r_n, fixed q, σ_n
        if call:
            p_bs = call_price(S, K, r_n, q, sigma_n, T)
        else:
            p_bs = put_price(S, K, r_n, q, sigma_n, T)

        # Weight = e^{-λ T} (λ T)^n / n!
        price += weight * p_bs

        # Update for next n
        if n < n_terms:
            fact *= (n + 1)  # (n+1)! = n! * (n+1)
            weight = np.exp(-pois_mean) * (pois_mean ** (n + 1)) / fact

    return float(price)




# ---------- SABR ----------

#  S0=St[i], K=strike[i], T=Ts[i]/252,r=r, q=self.q,
#                     sigma0=self.Vt[i], rho = self.rho, nu= self.nu

def sabr_call_price(
    S0: float,           # spot price
    K: float,            # strike
    T: float,            # time to maturity in years
    r: float,            # risk-free rate
    q: float = 0.0,      # dividend yield
    sigma0: float = 0.2, # initial volatility (α in some notations)
    rho: float = -0.4,   # correlation between spot and vol
    nu: float = 0.5      # vol-of-vol
) -> float:
    """
    Prices a European Call option under SABR model (β=1 case) using
    Hagan et al. (2002) implied volatility approximation + Black-Scholes formula.
    
    Parameters:
    -----------
    S0     : float - Current spot price
    K      : float - Strike price
    T      : float - Time to maturity (in years)
    r      : float - Risk-free interest rate (continuous)
    q      : float - Dividend yield (continuous), default=0
    sigma0 : float - SABR initial volatility parameter α
    beta   : float - Usually fixed to 1.0 for this formula
    rho    : float - Correlation between forward and volatility (-1 < rho < 1)
    nu     : float - Volatility of volatility parameter
    
    Returns:
    --------
    float - Call option price
    """
    
    # Forward price
    F = S0 * np.exp((r - q) * T)
    
    # Special case: ATM (very small z)
    if abs(F - K) < 1e-10 or T < 1e-12:
        # Very close to ATM → σ_B ≈ σ₀ * (1 + correction)
        z = 0.0
        sigma_B = sigma0 * (1 + (rho * nu * sigma0 / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T)
    else:
        # Hagan's variables
        z = nu / sigma0 * np.log(F / K)
        
        # x(z) function
        temp = np.sqrt(1 - 2 * rho * z + z**2) + z - rho
        x_z = np.log(temp / (1 - rho))
        
        # Avoid division by zero / numerical issues when z ≈ 0
        if abs(z) < 1e-8:
            sigma_B = sigma0
        else:
            sigma_B = sigma0 * (z / x_z) * (1 + (rho * nu * sigma0 / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T)
    
    # Black-Scholes d1, d2 with implied vol σ_B
    if sigma_B * np.sqrt(T) < 1e-10:
        d1 = d2 = np.inf if F > K else -np.inf
    else:
        d1 = (np.log(S0 / K) + (r-q+(sigma_B**2)/2) * T) / (sigma_B * np.sqrt(T))
        d2 = d1 - sigma_B * np.sqrt(T)
    
    # Black-Scholes call price
    discount = np.exp(-(r-q) * T)
    call_price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    
    return call_price

