# rs_gbm_analytical.py
# Analytical European call price under 2-state Regime-Switching GBM (RS-GBM)
# Based on Appendix D code (pp. 108–110) of the uploaded thesis.

import numpy as np
from math import log, pi
from scipy.integrate import quad

# ---------- helpers: mean/variance of log S(T) conditional on occupation time ----------
def mu(S0, d0, d1, r, sig0, sig1, T, t):
    """
    Mean of log S(T) given that the chain spends 't' in state 0 (and T - t in state 1).
    S0 : initial spot
    d0,d1 : dividend yields in states 0 and 1 (use 0 if none)
    r  : risk-free rate
    sig0,sig1 : volatilities in states 0 and 1
    T  : maturity
    t  : occupation time in state 0 used by the analytical formula
    """
    return log(S0) + (d1 - d0 - 0.5*(sig0**2 - sig1**2)) * t + (r - d1 - 0.5*sig1**2) * T

def var(sig0, sig1, T, t):
    """
    Variance of log S(T) under RS-GBM given occupation time 't' in state 0.
    """
    return (sig0**2 - sig1**2) * t + sig1**2 * T

# ---------- integrands for the two starting regimes ----------
def _integrand0(y, S0, d0, d1, r, sig0, sig1, T, l0, l1, K):
    # Appendix D uses two “mixing” points: t = T/3 and t = T for state 0 start
    m0T = mu(S0, d0, d1, r, sig0, sig1, T, T/3.0)
    mT  = mu(S0, d0, d1, r, sig0, sig1, T, T)
    v0T = var(sig0, sig1, T, T/3.0)
    vT  = var(sig0, sig1, T, T)

    a = (1.0/np.sqrt(2.0*pi*v0T)) * np.exp(-(np.log(y + K) - m0T)**2 / (2.0*v0T))
    b = (1.0/np.sqrt(2.0*pi*vT )) * np.exp(-(np.log(y + K) - mT )**2 / (2.0*vT ))

    # λ0 is the transition rate out of state 0
    return (y/(y+K)) * (a * (1.0 - np.exp(-l0*T)) + b * np.exp(-l0*T))

def _integrand1(y, S0, d0, d1, r, sig0, sig1, T, l0, l1, K):
    # Appendix D uses two points: t = T/3 and t = 0 for state 1 start
    m0T = mu(S0, d0, d1, r, sig0, sig1, T, T/3.0)
    m0  = mu(S0, d0, d1, r, sig0, sig1, T, 0.0)
    v0T = var(sig0, sig1, T, T/3.0)
    v0  = var(sig0, sig1, T, 0.0)

    a = (1.0/np.sqrt(2.0*pi*v0T)) * np.exp(-(np.log(y + K) - m0T)**2 / (2.0*v0T))
    b = (1.0/np.sqrt(2.0*pi*v0 )) * np.exp(-(np.log(y + K) - m0 )**2 / (2.0*v0 ))

    # λ1 is the transition rate out of state 1
    return (y/(y+K)) * (a * (1.0 - np.exp(-l1*T)) + b * np.exp(-l1*T))

# ---------- analytical prices when starting in regime 0 or 1 ----------
def call0(S0, d0, d1, r, sig0, sig1, T, l0, l1, K):
    """
    European call price when the initial regime is 0.
    """
    val, _ = quad(_integrand0, 0.0, np.inf, args=(S0, d0, d1, r, sig0, sig1, T, l0, l1, K))
    return np.exp(-r*T) * val

def call1(S0, d0, d1, r, sig0, sig1, T, l0, l1, K):
    """
    European call price when the initial regime is 1.
    """
    val, _ = quad(_integrand1, 0.0, np.inf, args=(S0, d0, d1, r, sig0, sig1, T, l0, l1, K))
    return np.exp(-r*T) * val

# ---------- convenience wrapper ----------
def call_rs_gbm(S0, K, r, sig0, sig1, l0, l1, T, d0=0.0, d1=0.0, init_state=0):
    """
    Wrapper that chooses call0 or call1 based on init_state (0 or 1).
    """
    if init_state == 0:
        return float(call0(S0, d0, d1, r, sig0, sig1, T, l0, l1, K))
    elif init_state == 1:
        return float(call1(S0, d0, d1, r, sig0, sig1, T, l0, l1, K))
    else:
        raise ValueError("init_state must be 0 or 1")

# ---------- quick self-test (matches Appendix D comments) ----------
if __name__ == "__main__":
    S0, K, r = 100.0, 90.0, 0.1
    d0 = d1 = 0.0
    sig0, sig1 = 0.2, 0.3
    l0 = l1 = 1.0
    T = 0.1

    c0 = call0(S0, d0, d1, r, sig0, sig1, T, l0, l1, K)
    c1 = call1(S0, d0, d1, r, sig0, sig1, T, l0, l1, K)
    print("call0 ≈", c0)  # Appendix D reference: ~21.0750372424
    print("call1 ≈", c1)  # Appendix D reference: ~22.0026453435
