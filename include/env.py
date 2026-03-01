from include.option_functions import calc_impl_volatility, calc_sabr_impl_volatility
import include.option_functions as option_functions
import numpy as np
import pandas as pd
import include.data_keeper as data_keeper
import include.simulation as simulation
import random
from datetime import datetime
from scipy.stats import norm
from include.settings import getSettings


class Env:
    def __init__(self, s=getSettings()):
        self.sim = simulation.Simulator(s['process'], periods_in_day=s['D'])
        self.s = s

        # ----------------------

        # ADD: seed + RNGs
        self._seed = int(self.s.get('seed', 12345))
        self.rng = np.random.RandomState(self._seed)   # use this instead of np.random
        self.py_rng = random.Random(self._seed)        # if you need Python's random

        # If your Simulator supports seeding, pass it now:
        try:
            self.sim.reseed(self._seed)
        except AttributeError:
           
            pass
        # -------- core process & simulator ----------
        self.process = self.s['process']                      # 'GBM' | 'Heston' | 'MJD' | 'RS_GBM' | 'Real'
        self.D       = self.s['D']                            # periods per day
        self.steps   = self.s['n_steps']
        self.sim     = simulation.Simulator(self.process, periods_in_day=self.D)

        # -------- generic env params ----------
        self.transaction_cost = self.s['transaction_cost']
        self.kappa            = self.s['kappa']
        self.reward_exponent  = self.s['reward_exponent']
                                 # diffusion vol for GBM/MJD baseline
        self.q                = self.s.get('q', 0.0)          # dividend yield
        self.r                = self.s.get('r', 0.0)          # risk-free (used in pricing)
        self.kappa_heston     = self.s.get('kappa_heston', 0.0)
        self.theta = self.s.get('theta', 0.0)
        self.sigma_vol_vol = self.s.get('sigma', 0.0)
        self.rho = self.s.get('rho', 0.0)
        self.v0 = self.s.get('v0', 0.0)
        self.SIGMA_GBM = 0.25
        # -------- MJD params (define even if not used to prevent AttributeError) ----------
        # Two notations appear in literature; keep both so either can be filled from JSON:
        # - (lam, m, v): J = lognormal(m, v^2) multiplier    (used by our internet_variant pricing)
        self.lam  = self.s.get('lam', 0.0)
        self.m    = self.s.get('m', 1.0)                      # mean jump multiplier (if used)
        self.v_jump    = self.s.get('v', 0.0)    
        self.SIGMA  =  self.s.get('SIGMA', 0.0)

        
        
         # -------- SABR Params -------------------------- 
        self.sigma0_sabr = self.s.get('sigma', 0.0) 
        self.nu= self.s.get('nu', 0.0) 
        self.rho = self.s.get('rho', 0.0) 
        self.q  = self.s.get('q', 0.0)          # dividend yield
        self.r  = self.s.get('r', 0.0)              # ln-jump stdev (if used)
        # self.SIGMA  =  self.v  

        # -------- RS_GBM params (2-regime) ----------
        # Vols and transition matrix; defaults keep attributes present even if unused.
        # Volatilities in regimes 0 and 1
        self.sigma0 = self.s.get('sigma0', 0.0) 
        self.sigma1 = self.s.get('sigma1', 0.0)

        # Continuous-time switching intensities (Poisson rates) out of each regime
        #   l0: rate of switching from regime 0 -> 1
        #   l1: rate of switching from regime 1 -> 0
        self.l0 = float(s.get('l0', 1.0))
        self.l1 = float(s.get('l1', 1.0))

        # Starting regime flag from config (your JSON uses 0 or 1; some files used 1/2).
        # Normalize to {0,1} internally.
        _sr = int(s.get('start_regime', 0))  # accept 0/1; if someone passes 1/2, map 1->0, 2->1
        if _sr in (0, 1):
            self.start_regime = _sr
        elif _sr == 2:
            self.start_regime = 1
        else:
            raise ValueError("start_regime must be 0 or 1 (or 1/2 old style).")

        # (Optional) stationary weights if you ever need them:
        den = self.l0 + self.l1
        if den > 0.0:
            # π0 = l1/(l0+l1), π1 = l0/(l0+l1)
            self.pi0 = self.l1 / den
            self.pi1 = self.l0 / den
        else:
            # Degenerate case: no switching; default all mass in start_regime
            self.pi0, self.pi1 = (1.0, 0.0) if self.start_regime == 0 else (0.0, 1.0)

        # -------- external data (only needed by some modes) ----------
        try:
            self.r_df = pd.read_csv('data/1yr_treasury.csv')
        except Exception:
            self.r_df = None
        try:
            self.heston_params = pd.read_csv('data/heston_params.csv')
        except Exception:
            self.heston_params = None

        # 'Real' data keeper if requested
        if self.process == 'Real':
            self.data_keeper = data_keeper.DataKeeper(self.steps)

        # -------- runtime state containers ----------
        self.data_set = pd.DataFrame()
        self.t, self.v, self.date_idx = 0, 0.0, 0
        self.option = {}                                   # option contract info
        self.S = []        

    def seed(self, seed: int):
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        # if you have a simulator/keeper, seed it too:
        if hasattr(self, 'data_keeper'): 
            try: self.data_keeper.seed(seed)
            except: pass
        return seed       
        
    def get_bs_delta(self):
        d1, _ = option_functions._d(self.option['S/K']*self.K, self.K, self.r, self.q, self.v, self.option['T']/365)
        return norm.cdf(d1)

    def __concat_state(self):
        return np.array([self.option['S/K'], self.option['T']/30, self.stockOwned, self.v])
    
    def __update_option(self):
        row = self.data_set.loc[self.t, :]

        spot = row['underlying_bid']
        P = 0.5 * (row['bid'] + row['ask'])
        self.expiry = row['expiration'][0:10]
        self.K = float(row['strike'])
        self.S[self.t] = spot
        self.cur_date = row['quote_datetime'][0:10]
        self.ticker = row['ticker']
        self.option['P'] = P

        # try:
        #     self.r = self.r_df.loc[self.r_df['Date'] == self.cur_date, '1y'].iloc[0]
        # except:
        #     # If r is missing, use previous (shouldn't happen)
        #     print("r missing:", self.cur_date)
        
        ttm = (datetime.strptime(self.expiry, '%Y-%m-%d') - \
            datetime.strptime(self.cur_date, '%Y-%m-%d')).days - (1 - (self.D - self.t%self.D) / self.D)

        self.option['T'] = ttm
        self.option['S/K'] = spot / self.K

        # if self.process == 'SABR':
        #     future_price = spot * np.exp((self.r-self.q)*(ttm/365))
        #     iv = calc_sabr_impl_volatility(future_price, self.K, ttm/365, self.v, self.rho, self.nu)
        # else: 
        #     iv = calc_impl_volatility(spot, self.K, self.r, self.q, ttm/365, P)
        iv = calc_impl_volatility(spot, self.K, self.r, self.q, ttm/365, P)
        # print(f'The IV is {iv}')
        # if iv > 1:
        #     print('H')
        # Sometimes impossible to solve IV, have to use the previous value
        if iv:
            self.v = iv
        
    def reset(self, testing = False, start_a = 0.0, start_b = 0.0):
        # Reset must be called when episode begins
        # testing indicates if empirical data should be used
        self.testing = testing
        self.t = 0
        self.S = np.zeros(self.steps + 1)
        
        self.stockOwned, self.b_stockOwned = start_a, start_b
        
        new_set = None
        
        if testing:
            self.data_set = self.data_keeper.next_test_set()
        else:
            if self.process == 'Real':
                self.data_set = self.data_keeper.next_train_set()
            else:
                while new_set is None:
                    dates = self.r_df['Date']
                    dates = dates[dates >='2013-01-01']
                    dates = sorted(dates.unique())[:-90]
                    # quote_datetime = np.random.choice(dates)
                    quote_datetime = self.rng.choice(dates)   # <- was np.random.choice


                    # try:
                    #     self.r = self.r_df.loc[self.r_df['Date'] == quote_datetime, '1y'].iloc[0]
                    # except:
                    #     self.r = 0.01
                    
                    if self.process == 'GBM':
                        self.sim.set_properties_gbm(self.r,self.SIGMA_GBM, self.q, .0)
                        T = self.steps + 1
                        dt = 1/(252*self.D)
                    elif self.process == 'MJD':
                        # Merton Jump–Diffusion
                        # Make sure these attrs exist on the env: self.lam (λ), self.muJ, self.sigJ, self.r, self.q
                        self.sim.set_properties_merton(
                            sigma=self.SIGMA,     # diffusion vol
                            lam=self.lam,         # jump intensity λ (per year)
                            m=self.m,         # mean of log-jump
                            v= self.v_jump ,       # stdev of log-jump
                            r=self.r,             # risk-free rate
                            q=self.q              # dividend yield
                        )
                        T  = self.steps + 1
                        dt = 1.0 / (252 * self.D)

                        # sigma, lam, m, v, r, q

                    elif self.process == 'SABR':
                        # Merton Jump–Diffusion
                        # Make sure these attrs exist on the env: self.lam (λ), self.muJ, self.sigJ, self.r, self.q
                        self.sim.set_properties_sabr(v = self.sigma0_sabr, nu= self.nu, rho = self.rho,r = self.r, q=self.q)    # dividend yield
                        
                        T  = self.steps + 1
                        dt = 1.0 / (252 * self.D)
                    elif self.process == 'RS_GBM':
                        # Regime-Switching GBM (2 states) with analytical-approach params
                        # Expect env to have: sigma0, sigma1, r, q, l0, l1, start_regime (0 or 1)
                        self.sim.set_properties_rs_gbm(
                            sigma0=self.sigma0,          # volatility in regime 0
                            sigma1=self.sigma1,          # volatility in regime 1
                            r=self.r,                    # risk-free rate
                            q=self.q,                    # dividend yield
                            l0=self.l0,                  # “rate/weight” for state 0 (analytical formula)
                            l1=self.l1,                  # “rate/weight” for state 1 (analytical formula)
                            start_regime=int(getattr(self, "start_regime", 0))  # 0 or 1
                        )
                        T  = self.steps + 1
                        dt = 1.0 / (252 * self.D)
                    else:
                        params = self.heston_params[self.heston_params['date'] == quote_datetime]
                        if params.empty:
                            continue

                        # v0 = params.iloc[0]['v0']
                        # kappa = params.iloc[0]['kappa']
                        # theta = params.iloc[0]['theta']
                        # sigma = params.iloc[0]['sigma']
                        # rho = params.iloc[0]['rho']

                        v0 = self.v0**2
                        kappa_heston = self.kappa_heston 
                        theta = self.theta
                        sigma = self.sigma_vol_vol
                        rho = self.rho

                        self.sim.set_properties_heston(v0, kappa_heston, theta, sigma, rho, self.q, self.r)
                        T = 5 # 5 ngay
                        dt = 35 # Moi ngay trade 7 lan 

                    self.sim.simulate(1.0, T, dt)
                    new_set = self.sim.return_set(.85, 1.15, quote_datetime, 15, 90, sorted(self.r_df['Date'].unique()), self.r)
                    # new_set = self.sim.return_set(1., 1., quote_datetime, 10, 10, sorted(self.r_df['Date'].unique()), self.r)

                    self.data_set = new_set
            
        self.__update_option()
        return self.__concat_state()

    def step(self, delta):
        # Step from T0 to T1
        def reward_func(pnl):
            # Reward scaled for clarity and small positive added
            pnl *= 100
            reward = 0.03 + pnl - self.kappa * (abs(pnl)**self.reward_exponent)
            return reward * 10
        
    #     def reward_func(pnl):
    #    # Paper’s objective: r_t = ΔPnL_t - κ |ΔPnL_t|^p
    #         return pnl - self.kappa * (abs(pnl) ** self.reward_exponent)
        
        infos = {'T':self.option['T'],
                'S/K':self.option['S/K']}
        
        infos['Date'] = self.cur_date
        infos['DateStep'] = self.t % self.D
        
        b_delta = self.get_bs_delta()
        
        #Linear transaction cost based on current (T0) face value and change in position
        t_cost = -abs(-delta - self.stockOwned) * self.S[self.t] * self.transaction_cost
        b_t_cost =  -abs(-b_delta - self.b_stockOwned) * self.S[self.t] * self.transaction_cost
        
        opt_old_price = self.option['P']
        
        self.t += 1
        
        self.__update_option()
        
        done = self.t >= self.steps

        opt_new_price = self.option['P']

        # PnL effect of underlying position
        pnl = -delta * (self.S[self.t] - self.S[self.t - 1])
        b_pnl = -b_delta * (self.S[self.t] - self.S[self.t - 1])
        
        # PnL effect of option price change and transaction cost
        pnl += (opt_new_price - opt_old_price) + t_cost
        b_pnl += (opt_new_price - opt_old_price) + b_t_cost        
        
        self.stockOwned = -delta
        self.b_stockOwned = -b_delta

        reward = reward_func(pnl)
        b_reward = reward_func(b_pnl)
        
        infos['B Reward'] = b_reward
        infos['A Reward'] = reward
        infos['A PnL'] = pnl
        infos['B PnL'] = b_pnl      
        infos['P0'] = opt_new_price  
        infos['P-1'] = opt_old_price
        infos['S0'] = self.S[self.t]
        infos['S-1'] = self.S[self.t - 1]
        infos['A Pos'] = self.stockOwned
        infos['B Pos'] = self.b_stockOwned
        infos['A TC'] = t_cost
        infos['B TC'] = b_t_cost
        infos['A PnL - TC'] = pnl - t_cost
        infos['B PnL - TC'] = b_pnl - b_t_cost
        infos['Expiry'] = self.expiry
        infos['v'] = self.v
        
        return self.__concat_state(), reward, done, infos
