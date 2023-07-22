import pandas as pd
from env.trading_env import TradingEnv
from agent.q_learner import Q_Agent
from utils import train_test_split
import numpy as np
import warnings
import random
import tqdm as tq

warnings.filterwarnings(action="ignore")

# Set data path 
path = "dataset/"
file_name = "NIFTY_50_Intraday.csv"
train_years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016"]
test_years = ["2017", "2018", "2019", "2020", "2021"]

train_df, test_df = train_test_split(path, file_name, train_years, test_years)

# Set file path to save 
save_file = "saved_csvs/trade_data.csv"

env_params = {
                "nifty_df": train_df,
                "save_file": save_file,
                "observation_dim": 100,
                "margin_pct": 0.15,
                "max_num_lots": 3,
                "lot_size": 25,
                "num_lots_held": 0
            }

env = TradingEnv(**env_params)

alpha = 0.25
gamma = 0.99
epsilon = 0.3
num_episodes = 10

rl_model = Q_Agent(
                        env, env.observation_space.n, env.action_space.n,
                        epsilon, alpha, gamma
                    )

# Train model using Q learning algorithm
for episode in range(num_episodes):
    print("Episode No. {:}".format(episode+1))
    state = env.reset()
    for i in tq.tqdm(range(len(train_df))):    
        action = rl_model.take_action(state)
        next_state, reward, eoc_indicator, _ = env.step(action)
        old_value = rl_model.q_table[state, action]
        next_action = rl_model.take_action(next_state)
        new_value = rl_model.q_table[next_state, next_action]
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * new_value)
        rl_model.q_table[state, action] = new_value
        state = next_state
        if eoc_indicator:
            state = env.reset()

model_path = "saved_models/"
with open(model_path + "q_table.npy", "wb") as fp:
    np.save(fp, rl_model.q_table)

np.savetxt(model_path + "q_table.txt", rl_model.q_table, delimiter=",")
