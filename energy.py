import numpy as np
import pandas as pd
import os
import argparse
from include.settings import getSettings, setSettings
from include.env import Env
from include.utility import StatePrepare, maybe_make_dirs
from include.actor_critic import ActorCritic
from testing import set_all_seeds
import json  # Need this for JSON export



def test_run_energy(env, actor_critic, scaler, state_size, episode_number, empirical):
    cur_state = env.reset(empirical)
    cur_state = scaler.transform(cur_state)
    cur_state = cur_state.reshape((1, state_size))
    infos = None
    
    stats = {'rewards':np.zeros(1), 'b rewards':np.zeros(1), 'pnl':np.zeros(1), 'b pnl':np.zeros(1)}
    i = 0

    energy_list = []

    done = False
    while not done:
        pred_action = actor_critic.act(cur_state)

        energy = actor_critic.actor.calculate_energy()
        action = np.clip( 0.5 * (pred_action[0] + 1), 0, 1) 

        new_state, reward, done, info = env.step(action)
      
        
        new_state = scaler.transform(new_state) 
        new_state = new_state.reshape((1, state_size))
        cur_state = new_state

        energy_list.append(energy)
        
       
    return np.mean(energy_list)

def test_load_energy(model_name):
    maybe_make_dirs()

    model_name, i = model_name.rsplit('_', 1)
    sim = 10000
    # sim = 10


    setSettings(model_name)
    seed = 12345       # add "seed" to your jsons
    set_all_seeds(seed)

    s = getSettings()
    env = Env(s)
    env.seed(seed)

    scaler = StatePrepare(env, 1, model_name)
    scaler.load(model_name)
    state_size = scaler.state_size
    
    actor_critic = ActorCritic(state_size,s=s)
    actor_critic.load('model/' + model_name + '_' + i)
    empirical = False 
    
   
    j = 0

    avg_energy = 0
    
    
    while j < sim:
        env.seed(seed + j)
        energy_episode = test_run_energy(env, actor_critic, scaler, state_size, j, empirical)
        avg_energy += energy_episode
        

        j += 1

    return avg_energy/sim

list_models = ['SABR_kappa2_snn_T2_38000',
               'SABR_kappa3_snn_T2_29000']

# list_models = ['Heston_kappa1_snn_T2_29000',
#                'Heston_kappa2_snn_T2_29000',
#                'Heston_kappa3_snn_T2_29000',
#                'MJD_kappa1_snn_T2_19000',
#                'MJD_kappa2_snn_T2_35000',
#                'MJD_kappa3_snn_T2_39000',
#                'SABR_kappa1_snn_T2_34000',
#                'SABR_kappa2_snn_T2_38000',
#                'SABR_kappa3_snn_T2_29000']

res = {}


ann_energy = 4.6 * (4 * 250 + 250**2 + 250 * 1) #(293250)

for model in list_models:
    avg_energy = test_load_energy(model)
    print(f'The model {model} and energy is {avg_energy}')
    res[model] = avg_energy

# Export the dictionary to a JSON file
with open('results.json', 'w') as f:
    json.dump(res, f, indent=4)  # indent=4 makes it human-readable