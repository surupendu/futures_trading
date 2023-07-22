import gym
from gym import spaces
import pandas as pd
from evaluate import Evaluate
import warnings
import numpy as np
import os

warnings.filterwarnings(action="ignore")

class TradingEnvCNN(gym.Env):
    """Custom Environment for Futures Trading"""
    metadata = {'render.modes': ['human']}

    def __init__(
                    self, nifty_df, save_file, action_dim, observation_dim,
                    margin_pct, max_num_lots, lot_size, num_lots_held, window_size,
                    lambda_
                ):
        super(TradingEnvCNN, self).__init__()
        self.nifty_df = self.eod(nifty_df)
        self.init_params(margin_pct, max_num_lots, lot_size, num_lots_held)
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.window_size = window_size
        self.lambda_ = lambda_
        self.episode = 0
        
        # Define action and observation space
        # Action space is discrete for DQN
        # Action space is continuous for PPO
        if self.model_name == "dqn":
            self.action_space = spaces.Discrete(2 * max_num_lots + 1)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))

        self.observation_space = spaces.Dict({
                                    "market_value": spaces.Box(low=-np.inf, high=np.inf,
                                                               shape=(self.window_size, self.observation_dim),
                                                               dtype=np.float64
                                                            ),
                                    "action": spaces.Box(low=-self.max_num_lots, high=self.max_num_lots,
                                                         shape=(1,),
                                                         dtype=np.float64
                                                        )
                                })
        
        self.save_file = save_file
        self.init_csv_file()

    def init_params(self, margin_pct, max_num_lots, lot_size, num_lots_held):
        """
            Initialize the parameters for futures trading
        """
        self.idx = 0
        self.balance = max_num_lots * lot_size * self.nifty_df.loc[0]["Close_1"]
        self.prev_close_price = self.nifty_df.loc[0]["Close_1"]
        self.initial_balance = self.balance
        self.margin_pct = margin_pct
        self.max_num_lots = max_num_lots
        self.lot_size = lot_size
        self.num_lots_held = num_lots_held
        self.max_num_lots_held = max_num_lots

    def init_csv_file(self):
        """
            Initalize the csv file for saving the output
        """
        self.dates = []
        self.times = []
        self.balances = []
        self.actions = []
        self.lots = []
        self.current_prices = []
        self.df = pd.DataFrame([], columns=["Date", "Time", "Balance", "Actions", "Lots", "Current Price"])

    def eod(self, nifty_df):
        """
            Calculate End of Day (EOD) indicator
        """
        nifty_df["Date_1"] = nifty_df["Date"].shift(-1)
        nifty_df["EOD"] = (nifty_df["Date_1"] != nifty_df["Date"])
        nifty_df = nifty_df.drop(columns=["Date_1"])
        return nifty_df        

    def insert_array_element(self, date, time, action, balance, current_price):
        self.dates.append(date)
        self.times.append(time)
        self.actions.append(action)
        self.balances.append(balance)
        self.lots.append(self.num_lots_held)
        self.current_prices.append(current_price)

    def insert_to_df(self):
        self.df["Date"] = self.dates
        self.df["Time"] = self.times
        self.df["Balance"] = self.balances
        self.df["Actions"] = self.actions
        self.df["Lots"] = self.lots
        self.df["Current Price"] = self.current_prices
        self.df.to_csv(self.save_file, index=False, mode='a', header=not os.path.exists(self.save_file))

    def execute_action(self, action, current_price, prev_price, eod_done, eoc_done):
        """
            Execute the number of lots to buy, sell or hold
        """
        
        # Control the agent from taking action which is more than maximum number of lots
        if ((self.num_lots_held+action) > self.max_num_lots_held) or ((self.num_lots_held+action) < -self.max_num_lots_held):
            action = 0

        # Check for End of Contract (EOC)
        if eoc_done == True:
            if self.num_lots_held < 0:
                action = abs(self.num_lots_held)
            elif self.num_lots_held > 0:
                action = -self.num_lots_held
            elif self.num_lots_held == 0:
                action = 0

        # Calculate contract value and margin
        contract_value = action * self.lot_size * current_price
        margin_value = self.margin_pct * contract_value
        # Update balance
        new_balance = self.balance - margin_value
        self.num_lots_held += action

        # Calculate Mark to Market
        if eod_done == True:
            new_balance += self.num_lots_held * self.lot_size * (current_price - self.prev_close_price)
            self.prev_close_price = current_price

        # Calculate reward
        reward = (self.lambda_) * action * (current_price - prev_price) + (1-self.lambda_) * (new_balance - self.balance)
        self.balance = new_balance
        return reward, action

    def step(self, action):
        '''
            Take action in environment
        '''
        
        if self.model_name == "ppo":
            # Get discrete value from continuous value
            action = action * self.max_num_lots
            action = action.astype(int)
            action = action[0]
        elif self.model_name == "dqn":
            # Get discrete value from the discrete action value
            action = action - self.max_num_lots
        
        prev_price = self.nifty_df.iloc[self.idx + self.window_size - 1]["Close_1"]

        self.idx += 1
        current_price = self.nifty_df.iloc[self.idx + self.window_size]["Close_1"]
        date = self.nifty_df.iloc[self.idx + self.window_size]["Date"]
        time = self.nifty_df.iloc[self.idx + self.window_size]["Time"]
        eod_indicator = self.nifty_df.iloc[self.idx + self.window_size]["EOD"]
        eoc_indicator = self.nifty_df.iloc[self.idx + self.window_size]["EOC"]
        
        reward, action = self.execute_action(action, current_price, prev_price, eod_indicator, eoc_indicator)
        market_values = np.array(self.nifty_df.iloc[self.idx:self.window_size + self.idx, 7:-3].values).astype("float64")
        
        action = np.array([action])
        
        next_state = {"market_value": market_values, "action": action}
        self.insert_array_element(date, time, action, self.balance, current_price)
        
        if eoc_indicator == True or self.balance < 0 or (self.idx + self.window_size) == len(self.nifty_df)-1:
            self.insert_to_df()
            eoc_indicator = True
            if self.idx + self.window_size == len(self.nifty_df)-1:
                eoc_indicator = True
                self.evaluate_model()

        return next_state, reward, eoc_indicator, {}

    def reset(self):
        '''
            Reset environment at end of episode or end of trading session
        '''
        if self.idx + self.window_size == len(self.nifty_df)-1:
            self.idx = 0
            self.balance = self.max_num_lots * self.lot_size * self.nifty_df.loc[self.idx]["Close_1"]
            self.prev_close_price = self.nifty_df.loc[self.idx]["Close_1"]
        self.num_lots_held = 0
        self.episode += 1
        market_values = np.array(self.nifty_df.iloc[self.idx:self.window_size + self.idx, 7:-3].values)
        action = np.array([0])
        next_state = {"market_value": market_values, "action": action}
        
        self.init_csv_file()
        return next_state

    def render(self, mode="human", close=False):
        return None

    def evaluate_model(self):
        '''
            Evaluate the model
        '''
        print("Initial Balance: {:}".format(self.initial_balance))
        print("Final Balance: {:}".format(self.balance))
        trade_df = pd.read_csv(self.save_file)
        evaluate_trade = Evaluate(trade_df, self.initial_balance)
        sharpe_ratio = evaluate_trade.calc_sharpe_ratio()
        sortino_ratio = evaluate_trade.calc_sortino_ratio()
        profit = evaluate_trade.calc_profit()
        max_draw_down, draw_down_duration = evaluate_trade.calc_max_drawdown()
        annualized_return = evaluate_trade.calc_annualized_return()
        volatility = evaluate_trade.calc_annualized_volatility()

        print("Sharpe Ratio: {:}".format(sharpe_ratio))
        print("Sortino Ratio: {:}".format(sortino_ratio))
        print("Total Profit: {:}".format(profit))
        print("Max Draw Down (%): {:}".format(max_draw_down))
        print("Max Draw Down duration (days): {:}".format(draw_down_duration))
        print("Return (%): {:}".format(annualized_return))
        print("Volatility (%): {:}".format(volatility))
