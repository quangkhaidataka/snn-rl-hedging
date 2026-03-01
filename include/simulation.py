import numpy as np
import pandas as pd
import QuantLib as ql
import random
import include.option_functions as option_functions

class Simulator():
    def __init__(self, process, periods_in_day = 1,seed=12345):
        self.process = process
        self.D = periods_in_day
        self.seed = int(seed)

        self.np_rng = np.random.RandomState(self.seed)
        self.py_rng = random.Random(self.seed)

    def reseed(self, seed):
        self.seed = int(seed)
        self.np_rng = np.random.RandomState(self.seed)
        self.py_rng = random.Random(self.seed)

    def set_properties_rs_gbm(self, *, sigma0, sigma1, l0, l1, r, q, start_regime=0):
        """
        Set parameters for a 2-state Regime-Switching GBM using continuous-time
        Markov chain intensities (ℓ₀, ℓ₁) instead of a transition matrix.

        Parameters
        ----------
        sigma0 : float
            Volatility in regime 0 (low/first regime).
        sigma1 : float
            Volatility in regime 1 (high/second regime).
        l0 : float
            Exit intensity from regime 0 (rate of 0 -> 1), ℓ₀ ≥ 0.
        l1 : float
            Exit intensity from regime 1 (rate of 1 -> 0), ℓ₁ ≥ 0.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        start_regime : int, optional
            Initial regime indicator: 0 or 1 (default: 0).

        Notes
        -----
        - This parameterization aligns with the analytical pricing code that
        uses (ℓ₀, ℓ₁) explicitly.
        - If you previously passed a discrete-time transition matrix P, convert
        it to intensities externally before calling this method.
        """
        # --- basic validation ---
        for name, val in (("sigma0", sigma0), ("sigma1", sigma1), ("l0", l0), ("l1", l1)):
            if float(val) < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        if start_regime not in (0, 1):
            raise ValueError("start_regime must be 0 or 1")

        # --- store parameters ---
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.l0 = float(l0)            # intensity 0 -> 1
        self.l1 = float(l1)            # intensity 1 -> 0
        self.r  = float(r)
        self.q  = float(q)

        # initial regime (0 or 1)
        self.start_regime = int(start_regime)

        # (Optional) convenience: current regime initialized to start_regime
        # if you simulate paths step-by-step and keep a regime state.
        self._regime = self.start_regime

    # No discrete-time transition matrix P is stored or used here by design.



        
    def set_properties_gbm(self, r,v, q, mu):
        self.v0 = v
        self.r = r
        self.q = q
        self.mu = mu

    def set_properties_heston(self, v0, kappa, theta, sigma, rho, q, r):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.q = q
        self.r = r

    def set_properties_merton(self, sigma, lam, m, v, r, q):
        """
        Set parameters for Merton Jump–Diffusion with lognormal jump sizes.

        J ~ LogNormal(muJ, sigJ^2)
        sigma : diffusion vol (per sqrt(year))
        lam   : Poisson jump intensity λ (per year)
        r, q  : risk-free rate, dividend yield
        """
        # store raw inputs
        self.sigma = float(sigma)     # diffusion σ
        self.v0    = float(sigma)     # keep v0 for consistency with your GBM code
        self.lam   = float(lam)
        self.m   = float(m)
        self.v  = float(v)
        self.r     = float(r)
        self.q     = float(q)

    def set_properties_sabr(self, v, nu, rho,r, q):
        self.v0 = v
        self.nu = nu
        self.rho = rho
        self.q = q
        self.r = r

            
    def simulate(self, S0, T = 252, dt = 1/252):
        if self.process == 'GBM':
            self._sim_gbm(S0, self.mu, self.v0, T, dt)
        elif (self.process == 'Heston') or (self.process == 'Real'):
            self._sim_heston(S0, self.v0, self.kappa, self.theta, self.sigma, self.rho, self.q, self.r, T, dt)
        elif self.process == 'MJD':  # or 'Merton'
            self._sim_merton_jump(
            S0, self.m, self.v0, self.lam,self.v, T, dt, self.r, self.q # set True if you want RN paths
            )
        elif self.process == 'SABR':
            self._sim_sabr(S0, self.v0, self.nu, self.rho, T, dt, self.r, self.q )

        elif self.process == 'RS_GBM':
            # Regime-Switching GBM with exponential holding times (rates l0, l1)
            # Requires set_properties_rs_gbm(sigma0=..., sigma1=..., l0=..., l1=..., r=..., q=..., start_regime=...)
            self._sim_rs_gbm(
                S0=S0,
                sigma0=self.sigma0,
                sigma1=self.sigma1,
                r=self.r,
                q0=self.q,
                q1=self.q,
                l0=self.l0,
                l1=self.l1,
                T=T,
                dt=dt,
                start_state=self.start_regime  # 0 or 1
            )
        else:
            raise ValueError(f"Unknown process: {self.process}")



    def _sim_gbm(self, S0, mu, stdev, T, dt):
        self.St = np.zeros(T)
        self.St[0] = S0
                
        for t in range(1, T):
            z = self.np_rng.normal()
            self.St[t] = self.St[t-1] * np.exp(mu * dt + stdev * np.sqrt(dt)*z)


    def _sim_rs_gbm(self, S0, r,                # risk-free
                q0, q1,                      # dividends in regime 0/1 (these are your d0,d1)
                sigma0, sigma1,              # vols in regime 0/1
                T, dt,                       # total steps (int) and step size (float)
                start_state=0,               # 0 or 1
                P=None,                      # 2x2 transition matrix per step (discrete-time)
                l0=None, l1=None,            # OR continuous-time rates: λ0 (0->1), λ1 (1->0)
                seed=None):
        """
        Simulate a 2-regime RS-GBM price path S_t and regime path Z_t.

        Dynamics in regime z ∈ {0,1}:
            dS/S = (r - q_z) dt + sigma_z dW

        Regime process:
        - If P is given (per-step), use P[z] = [p_{z1->0}, p_{z1->1}] (rows sum to 1).
        - Else if (l0,l1) are given, use p_{01}=1-exp(-l0*dt), p_{10}=1-exp(-l1*dt).
        - Exactly one of {P} or {(l0,l1)} should be provided.

        Stores:
        self.St     : np.ndarray shape (T,), price path
        self.Zt     : np.ndarray shape (T,), regime path (0/1)
        self.t_occ0 : total occupation time in regime 0 (≈ count(Z_t=0)*dt)
        """
        rng = np.random.default_rng(seed)

        # --- per-step switch probs ---
        if P is not None:
            P = np.asarray(P, dtype=float)
            assert P.shape == (2, 2), "P must be 2x2"
            # Row i: [p(i->0), p(i->1)]
            # (Allow either row-stochastic with p(i->0)+p(i->1)=1, or specify only p(i->1) in [0,1])
            # We'll normalize rows to be safe:
            P = P / P.sum(axis=1, keepdims=True)
            p01 = P[0, 1]
            p10 = P[1, 0]
        else:
            assert (l0 is not None) and (l1 is not None), "Provide either P or (l0,l1)."
            # continuous-time → discrete step switching probabilities
            p01 = 1.0 - np.exp(-float(l0) * dt)  # 0 -> 1
            p10 = 1.0 - np.exp(-float(l1) * dt)  # 1 -> 0

        # --- pre-alloc ---
        self.St = np.zeros(T, dtype=float)
        self.Zt = np.zeros(T, dtype=int)

        self.St[0] = float(S0)
        self.Zt[0] = int(start_state)

        # --- simulate ---
        for t in range(1, T):
            z = self.Zt[t-1]                      # previous regime
            # drift/vol for this step
            q  = q0 if z == 0 else q1
            sig= sigma0 if z == 0 else sigma1

            # GBM Euler–Maruyama update in log space (exact step for GBM):
            dW = rng.normal()
            self.St[t] = self.St[t-1] * np.exp((r - q - 0.5*sig*sig)*dt + sig*np.sqrt(dt)*dW)

            # Regime switch
            u = rng.random()
            if z == 0:
                self.Zt[t] = 1 if (u < p01) else 0
            else:
                self.Zt[t] = 0 if (u < p10) else 1

        # --- occupation time in state 0 (optional, useful to link to analytical formula) ---
        self.t_occ0 = float((self.Zt == 0).sum() * dt)

    def _sim_sabr(self,S0, sigma0, nu, rho, T, dt, r=0.0, q=0.0):
    
        # Cast to floats
        S0 = float(S0)
        sigma0 = float(sigma0)
        nu = float(nu)
        rho = float(rho)
        dt = float(dt)
        r = float(r)
        q = float(q)

        # Number of steps
        n_steps = int(T)
        size= n_steps
        sqrt_dt = np.sqrt(dt)

        # Generate independent Gaussians
        Z1 = np.random.normal(size=size)
        Z2 = np.random.normal(size=size)

        # Correlated Brownian increments
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        # Initialize paths
        St = np.zeros(n_steps)
        sigmat = np.zeros(n_steps)

        St[0] = S0
        sigmat[0] = sigma0

        # Simulation loop (necessary due to state-dependent diffusion)
        for i in range(n_steps-1):
            # Exact update for σ_t (lognormal)
            sigmat[i+1] = sigmat[i] * np.exp(-0.5 * nu**2 * dt + nu * dW2[i])

            # Log-Euler update for S_t (preserves positivity)
            mu_dt = (r - q) * dt
            St[i+1] = St[i] * np.exp(mu_dt - 0.5 * sigmat[i]**2 * dt + sigmat[i] * dW1[i])

        self.St = St  # Optio
        self.Vt = sigmat


    # include/simulation.py  -- replace your function with this version

    def _sim_merton_jump(self, S0, m, sigma, lam, v, T, dt, r=0.0, q=0.0):
        """
        Simulate Merton Jump-Diffusion with a *seeded* RNG (self.np_rng).

        Inputs:
        T  : number of time steps (int)
        dt : step size in years (float), e.g. 1/(252*D)

        Uses:
        z ~ N(0,1) from self.np_rng.normal()
        K ~ Poisson(λ dt) from self.np_rng.poisson()
        sum_logJ ~ N(K*(m), K v^2) via K*m + sqrt(K)*v * self.np_rng.normal()
        """

        sigma = float(sigma)
        lam   = float(lam)
        m     = float(m)
        v     = float(v)
        dt    = float(dt)
        r     = float(r)
        q     = float(q)

        # Compute the exact compensator k = exp(m + v^2 / 2) - 1
        k = np.exp(m + 0.5 * v**2) - 1.0

        n_steps = int(T) - 1
        size = (n_steps, 1)

        # Jump part: for each step, K ~ Poisson(lam*dt), then log_jump = m*K + v*sqrt(K)*N(0,1)
        Ks = self.np_rng.poisson(lam * dt, size=size)
        normals_jump = self.np_rng.normal(size=size)
        poi_rv = (m * Ks + v * np.sqrt(Ks) * normals_jump).cumsum(axis=0)

        # Diffusion part: drift = (r - q - lam*k - sigma^2/2)*dt + sigma*sqrt(dt)*N(0,1)
        drift = (r - q - lam * k - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * self.np_rng.normal(size=size)
        geo = np.cumsum(drift + diffusion, axis=0)

        path = np.exp(geo + poi_rv) * S0
        path = path.reshape(-1)

        self.St = np.concatenate(([S0], path))

    
    def _sim_heston(self, S0, v0, kappa, theta, sigma, rho, q, r, T, dt):
        r_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), r, ql.Actual360()))
        q_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), q, ql.Actual360()))
        s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        process = ql.HestonProcess(r_handle, q_handle, s0_handle, v0, kappa, theta, sigma, rho)

        # Build a time grid with the intended number of steps
        # (In your code T=5 (days) and dt=35 (#steps))
        time_grid = ql.TimeGrid(T / 365.25, int(dt))
        n_steps = len(time_grid) - 1
        dim = process.factors()                # Heston -> 2 factors

        # ---- Seeded RNG chain (no SeedGenerator needed) ----
        urng  = ql.UniformRandomGenerator(int(self.seed))                # <-- seeded
        usg   = ql.UniformRandomSequenceGenerator(dim * n_steps, urng)   # sequence of uniforms
        grsg  = ql.GaussianRandomSequenceGenerator(usg)                  # gaussians

        seq = ql.GaussianMultiPathGenerator(process, time_grid, grsg, False)

        path = seq.next()
        values = path.value()
        St, Vt = values
        self.St = np.array([x for x in St])
        self.Vt = np.array([x for x in Vt])



    def getS(self):
        return self.St
    
    def return_set(self, strike_min, strike_max, quote_datetime, min_exp, max_exp, datearray, r):
        # Returns a simulated which looks similar to DataKeeper's sets 
        
        # strike = random.uniform(strike_min, strike_max)
        strike = self.py_rng.uniform(strike_min, strike_max)
        strike = [strike] * len(self.St)
        
        # exp = random.randint(min_exp, max_exp)
        exp = self.py_rng.randint(min_exp, max_exp)
        expiration = datearray[datearray.index(quote_datetime) + int(exp)]
        expiration = [expiration] * len(self.St)
        
        quote_datetimes = []
        
        i = 0
        while len(quote_datetimes) < len(self.St):
            temp = [datearray[datearray.index(quote_datetime) + int(i)]] * self.D
            quote_datetimes += temp
            i = i + 1
            
        quote_datetimes = quote_datetimes[:len(self.St)]

        St = self.St / self.St[0]
        
        Ts = exp - np.arange(0, len(self.St)/(1/self.D), 1/self.D)

        df = pd.DataFrame()
        df['underlying_bid'] = St
        df['expiration'] = expiration
        df['strike'] = strike
        df['quote_datetime'] = quote_datetimes
        df['ticker'] = 'simulated'
        
        prices = []
        
        for i in range(len(self.St)):
            if self.process == 'GBM':
                price = option_functions.call_price(St[i], strike[i], r, self.q, self.v0, Ts[i]/252)
            elif (self.process == 'Heston') or (self.process == 'Real'):
                price = option_functions.heston_price(St[i], strike[i], r, self.q, self.theta,
                    self.kappa, self.sigma, self.rho, self.Vt[i], expiration[i], quote_datetimes[i])
                
            elif self.process == 'MJD':
                # Merton Jump–Diffusion (internet variant weights with m*λ)
                # self.v0 = diffusion sigma, self.lam = λ, self.m = m, self.v = v
                price = option_functions.mjd_price(
                    S=St[i], K=strike[i], r=r, q=self.q,
                    sigma=self.v0, T=Ts[i]/252,
                    lam=self.lam, m=self.m, v=self.v
                )
  
            elif self.process == 'SABR':
                # Merton Jump–Diffusion (internet variant weights with m*λ)
                # self.v0 = diffusion sigma, self.lam = λ, self.m = m, self.v = v
                price = option_functions.sabr_call_price(
                    S0=St[i], K=strike[i], T=Ts[i]/252,r=r, q=self.q,
                    sigma0=self.Vt[i], rho = self.rho, nu= self.nu
                )
            

            elif self.process == 'RS_GBM':
                # Regime-Switching Black–Scholes (analytical, Appendix D-style)
                # Required attributes on `self`:
                #   self.sigma0, self.sigma1    -> vols in regime 0 & 1
                #   self.l0, self.l1            -> regime “intensity” params (set to 1.0 if unused)
                #   self.r, self.q              -> risk-free rate, dividend yield
                #   self.start_regime           -> 0 or 1 (starting regime)
                T_years = max(Ts[i] / 252.0, 0.0)

                    # start in regime 0
                price = option_functions.rs_gbm_call_start0(
                    S0=St[i],
                    q0=self.q, q1=self.q,            # same dividend yield in both regimes
                    r=self.r,
                    sig0=self.sigma0,
                    sig1=self.sigma1,
                    T=T_years,
                    l0=getattr(self, "l0", 1.0),
                    l1=getattr(self, "l1", 1.0),
                    K=strike[i]
                )
                

            else:
                raise ValueError(f"Unknown process: {self.process}")
            
            prices.append(price)

        df['bid'] = prices
        df['ask'] = prices
        
        return df