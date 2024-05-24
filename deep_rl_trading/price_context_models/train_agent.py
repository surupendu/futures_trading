from trading_env.trading_env import Env_CNN_News_Price
from agent.ppo_agent import PPO_Agent
import pandas as pd
import warnings
import ast
import json
import argparse

warnings.filterwarnings(action="ignore")

"""
    Code to train the PPO-based RL agent
    use python train_agent.py [-h] to view the options available
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--text_representation', type=str,
                        help='text representation scheme',
                        choices=["bert", "finbert", "finbert_senti"])
    parser.add_argument('--text_sentiment', type=str,
                        help='use text sentiment along with representation scheme',
                        choices=["True", "False"])
    args = parser.parse_args()

    # Select config file based on text representation scheme
    if args.text_representation == "bert":
        config = json.load(open("configs/bert.json", "r"))
        approach = "bert"
    elif args.text_representation == "finbert":
        config = json.load(open("configs/finbert.json", "r"))
        approach = "finbert"
    elif args.text_representation == "finbert_senti":
        config = json.load(open("configs/finbert_senti.json", "r"))
        approach = "finbert_senti"

    # Read the train data
    train_df = pd.read_csv(config["path"] + "train_df_news.csv")

    news_df = train_df["Titles"]
    news_df = news_df.apply(lambda x: ast.literal_eval(x))
    train_df = train_df[train_df.columns[:-1]]

    save_file = config["save_file_path"] + "trade_data_{:}.csv".format(config["model_name"])

    # Initialize the parameters of the environment
    env_parameters = {
                        "nifty_df": train_df,
                        "news_df": news_df,
                        "save_file": save_file,
                        "action_dim": config["action_dim"],
                        "max_num_lots": config["max_num_lots"],
                        "lot_size": config["lot_size"],
                        "num_lots_held": config["num_lots_held"],
                        "margin_pct": config["margin_pct"],
                        "window_size": config["window_size"],
                        "num_news_articles": config["num_news_articles"],
                        "language_model": config["language_model"]
                    }

    observation_dim = len(train_df.columns[7:-2])
    env_parameters["observation_dim"] = observation_dim

    # Intialize the environment
    env = Env_CNN_News_Price(**env_parameters)

    model_parameters = {
                            "policy": config["policy"],
                            "env": env,
                            "learning_rate": config["learning_rate"],
                            "n_steps": config["n_steps"],
                            "batch_size": config["batch_size"],
                            "ent_coef": config["ent_coef"],
                            "n_epochs": config["n_epochs"],
                            "seed": 42,
                            "verbose": 1,
                            "approach": approach
                        }

    # Intialize the PPO agent and train the model
    rl_agent = PPO_Agent(config["model_name"], **model_parameters)
    rl_agent.train_model(total_timesteps=len(train_df) * config["global_num_epochs"])

    # Save the model
    save_model_path = "save_model/"
    file_name = "{:}_{:}".format("ppo", config["language_model"])
    rl_agent.save_model(save_model_path, file_name)
