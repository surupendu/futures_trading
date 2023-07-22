from env.trading_env import TradingEnv
from agent.q_learner import Q_Agent
import numpy as np
import pandas as pd
import tqdm as tq
from utils import train_test_split
import warnings

warnings.filterwarnings(action="ignore")

path = "dataset/"
file_name = "NIFTY_50_Intraday.csv"
train_years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016"]
test_years = ["2017", "2018", "2019", "2020", "2021"]

train_df, test_df = train_test_split(path, file_name, train_years, test_years)

env_params = {
                "observation_dim": 100,
                "margin_pct": 0.15,
                "max_num_lots": 3,
                "num_lots_held": 0
            }

for test_year in test_years:
    print("--------------Year: {:}--------------".format(test_year))
    save_file = "saved_csvs/trade_data_{:}.csv".format(test_year)
    max_num_lots = 3
    if test_year in ["2018", "2019", "2020", "2021"]:
        lot_size = 75
    else:
        lot_size = 25
    test_df_1 = test_df[test_df["Date"].str.contains(test_year)]
    test_df_1.reset_index(drop=True, inplace=True)

    env_params["nifty_df"] = test_df_1
    env_params["lot_size"] =  lot_size
    env_params["save_file"] = save_file
    env = TradingEnv(**env_params)

    rl_model = Q_Agent(env, env.observation_space.n, env.action_space.n)

    model_path = "saved_models/"
    with open(model_path + "q_table.npy", "rb") as fp:
        rl_model.q_table = np.load(fp)

    state = env.reset()
    action = rl_model.take_action_(state)
    for i in tq.tqdm(range(len(test_df_1))):
        next_state, reward, eoc_indicator, _ = env.step(action)    
        next_action = rl_model.take_action_(next_state)
        
        state = next_state
        action = next_action

        if eoc_indicator:
            state = env.reset()
