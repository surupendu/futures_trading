from trading_envs.trading_env_mlp import TradingEnvMLP
from trading_envs.trading_env_cnn import TradingEnvCNN
from agent.ppo_agent import PPO_Agent
from agent.dqn_agent import DQN_Agent
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings(action="ignore")

'''
    Set use_news to True use news sentiment,
    Set model_name to dqn or ppo
    Set approach to mlp or cnn
'''
use_news = True
model_name = "ppo"
approach = "cnn"

if use_news and approach == "cnn":
    # Set 1 hr sentiment for PPO_CNN_PT
    # Set data path
    data_path = "dataset/sentiment_data_1hr/"
elif use_news and approach == "mlp":
    # Set 5 hr sentiment for PPO_PT
    # Set data path
    data_path = "dataset/sentiment_data_5hr/"
else:
    # Set 1 hr and 5 hr sentiment for PPO_P, PPO_CNN_P, DQN_P, DQN_CNN_P
    data_path = "dataset/numerical_data/"

test_df = pd.read_csv(path + "test_df.csv")
lambda_ = 0.85

save_model_path = "trained_model/"

# Load model name
if use_news:
    model_file_name = "{:}_{:}_policy_news".format(model_name, approach)
else:
    model_file_name = "{:}_{:}_policy".format(model_name, approach)

test_years = ["2017", "2018", "2019", "2020", "2021"]

for test_year in test_years:
    save_file = "saved_csvs/trade_data_{:}.csv".format(test_year)
    max_num_lots = 3
    if test_year in ["2018", "2019", "2020", "2021"]:
        lot_size = 75
    else:
        lot_size = 25
    
    test_df_1 = test_df[test_df["Date"].str.contains(test_year)]
    test_df_1.reset_index(drop=True, inplace=True)

    action_dim = 1

    env_parameters = {
                        "nifty_df": test_df_1,
                        "save_file": save_file,
                        "action_dim": action_dim,
                        "max_num_lots": max_num_lots,
                        "lot_size": lot_size,
                        "num_lots_held": 0,
                        "margin_pct": 0.15,
                        "lambda_": lambda_,
                        "model_name": model_name
                    }
    
    if approach == "mlp":
        '''
            Set parameters for MLP
        '''
        observation_dim = len(test_df.columns[7:-2]) + 1
        env_parameters["observation_dim"] = observation_dim
        env = TradingEnvMLP(**env_parameters)
        policy = "MlpPolicy"
    
    if approach == "cnn":
        '''
            Set parameters for CNN
        '''
        env_parameters["window_size"] = 5
        observation_dim = len(test_df.columns[7:-2])
        env_parameters["observation_dim"] = observation_dim
        env = TradingEnvCNN(**env_parameters)
        policy = "MultiInputPolicy"
    
    # Run PPO model
    if model_name == "ppo":
        batch_size = 128
        learning_rate = 0.0002
        ent_coef = 0.02
        n_epochs = 5
        n_steps = 50
        model_parameters = {
                                "policy": policy,
                                "env": env,
                                "learning_rate": learning_rate,
                                "n_steps": n_steps,
                                "batch_size": batch_size,
                                "ent_coef": ent_coef,
                                "n_epochs": n_epochs,
                                "seed": 42,
                                "verbose": 1,
                            }
        rl_agent = PPO_Agent(model_name, **model_parameters)

    # Run DQN model
    if model_name == "dqn":
        batch_size = 64
        learning_rate = 0.0002
        buffer_size = 1000
        learning_starts = 300
        train_freq = 50
        gradient_steps = -1
        target_update_interval = 100
        seed = 42
        global_num_epochs = 3
        model_parameters = {
                                "policy": policy,
                                "env": env,
                                "batch_size": batch_size,
                                "learning_rate": learning_rate,
                                "buffer_size": buffer_size,
                                "learning_starts": learning_starts,
                                "train_freq": train_freq,
                                "gradient_steps": gradient_steps,
                                "target_update_interval": target_update_interval,
                                "seed": seed,
                                "verbose": 1,
                            }
        rl_agent = DQN_Agent(model_name, **model_parameters)

    rl_agent.load_model(save_model_path, model_file_name)
    print("--------------Year: {:}--------------".format(test_year))
    rl_agent.test_model(env)
    print("-------------------------------------")
