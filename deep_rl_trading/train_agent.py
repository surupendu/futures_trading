from trading_envs.trading_env_mlp import TradingEnvMLP
from trading_envs.trading_env_cnn import TradingEnvCNN
from agent.ppo_agent import PPO_Agent
from agent.dqn_agent import DQN_Agent
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action="ignore")

'''
    Set use_news to True use news sentiment,
    Set model_name to dqn or ppo
    Set approach to mlp or cnn
'''
use_news = True
model_name = "ppo"
approach = "cnn"

'''
    Set max_num_lots, lot_size, action_dim, lambda_
'''
max_num_lots = 3
lot_size = 25
action_dim = 1
lambda_ = 0.85

if model_name == "ppo":
    '''
        Set parameters for running PPO
    '''
    batch_size = 128
    learning_rate = 0.0002
    ent_coef = 0.02
    n_epochs = 9
    n_steps = 50
    global_num_epochs = 1

if model_name == "dqn":
    '''
        Set parameters for running DQN
    '''
    batch_size = 32
    learning_rate = 0.0012
    buffer_size = 10000
    learning_starts = 1000
    train_freq = 3500
    gradient_steps = -1
    target_update_interval = 10000
    seed = 42
    global_num_epochs = 1


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

train_df = pd.read_csv(data_path + "train_df.csv")

saved_path = "saved_csvs/"
save_file = saved_path + "trade_data_{:}.csv".format(model_name)

env_parameters = {
                    "nifty_df": train_df,
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
    observation_dim = len(train_df.columns[7:-2]) + 1
    env_parameters["observation_dim"] = observation_dim
    env = TradingEnvMLP(**env_parameters)
    policy = "MlpPolicy"

if approach == "cnn":
    '''
        Set parameters for CNN
    '''
    env_parameters["window_size"] = 5
    observation_dim = len(train_df.columns[7:-2])
    env_parameters["observation_dim"] = observation_dim
    env = TradingEnvCNN(**env_parameters)
    policy = "MultiInputPolicy"

# Run PPO model
if model_name == "ppo":
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
    rl_agent.train_model(total_timesteps=len(train_df) * global_num_epochs)

# Run DQN model
if model_name == "dqn":
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
    rl_agent.train_model(total_timesteps=len(train_df) * global_num_epochs)

# Save models to path
save_model_path = "trained_model/"

if use_news:
    file_name = "{:}_{:}_policy_news".format(model_name, approach)
else:
    file_name = "{:}_{:}_policy".format(model_name, approach)

rl_agent.save_model(save_model_path, file_name)
