import gym
from gym import spaces
import pandas as pd
from evaluate import Evaluate
import warnings
import numpy as np
import os

warnings.filterwarnings(action="ignore")

class TradingEnv(gym.Env):
    """Custom Environment for Futures Trading"""
    metadata = {'render.modes': ['human']}

    def __init__(
                    self, nifty_df, save_file, observation_dim,
                    margin_pct, max_num_lots, lot_size, num_lots_held
                ):
        super(TradingEnv, self).__init__()
        self.nifty_df = self.eod(nifty_df)
        self.init_params(margin_pct, max_num_lots, lot_size, num_lots_held)
        self.observation_dim = observation_dim
        self.episode = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2 * max_num_lots + 1)
        self.observation_space = spaces.Discrete(self.observation_dim)
        
        self.save_file = save_file
        self.init_csv_file()

    def init_params(self, margin_pct, max_num_lots, lot_size, num_lots_held):
        '''
            Initialize the parameters to be used for futures trading
        '''
        self.idx = 0
        self.balance = max_num_lots * lot_size * self.nifty_df.loc[0]["Close"]
        self.prev_close_price = self.nifty_df.loc[0]["Close"]
        self.initial_balance = self.balance
        self.margin_pct = margin_pct
        self.max_num_lots = max_num_lots
        self.lot_size = lot_size
        self.num_lots_held = num_lots_held
        self.max_num_lots_held = max_num_lots

    def init_csv_file(self):
        # Form a csv file
        self.dates = []
        self.times = []
        self.balances = []
        self.actions = []
        self.lots = []
        self.current_prices = []
        self.df = pd.DataFrame([], columns=["Date", "Time", "Balance", "Actions", "Lots", "Current Price"])

    def eod(self, nifty_df):
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
        if ((self.num_lots_held+action) > self.max_num_lots_held) or ((self.num_lots_held+action) < -self.max_num_lots_held):
            action = 0
        
        if eoc_done == True:
            if self.num_lots_held < 0:
                action = abs(self.num_lots_held)
            elif self.num_lots_held > 0:
                action = -self.num_lots_held
            elif self.num_lots_held == 0:
                action = 0
        
        contract_value = action * self.lot_size * current_price
        self.margin_value = self.margin_pct * contract_value
        new_balance = self.balance - self.margin_value
        self.num_lots_held += action

        # Calculate Mark to Market
        if eod_done == True:
            new_balance += self.num_lots_held * self.lot_size * (current_price - self.prev_close_price)
            self.prev_close_price = current_price

        reward = 0.85 * action * (current_price - prev_price) + 0.15 * (new_balance - self.balance)
        
        self.balance = new_balance
        return reward, action

    def step(self, action):
        # Execute one time step within the environment
        action = action - self.max_num_lots
        
        prev_price = self.nifty_df.iloc[self.idx]["Close"]

        self.idx += 1
        current_price = self.nifty_df.iloc[self.idx]["Close"]
        date = self.nifty_df.iloc[self.idx]["Date"]
        time = self.nifty_df.iloc[self.idx]["Time"]
        next_state = self.nifty_df.iloc[self.idx]["State"]
        eod_indicator = self.nifty_df.iloc[self.idx]["EOD"]
        eoc_indicator = self.nifty_df.iloc[self.idx]["EOC"]
        
        reward, action = self.execute_action(action, current_price, prev_price, eod_indicator, eoc_indicator)
        action = np.array([action])

        self.insert_array_element(date, time, action, self.balance, current_price)

        if eoc_indicator == True or self.balance < 0 or self.idx == len(self.nifty_df)-1:
            self.insert_to_df()
            eoc_indicator = True
            if self.idx == len(self.nifty_df)-1:
                eoc_indicator = True
                self.evaluate_model()

        return next_state, reward, eoc_indicator, {}

    def reset(self):
        if self.idx == len(self.nifty_df)-1:
            self.idx = 0
            self.balance = self.max_num_lots * self.lot_size * self.nifty_df.loc[self.idx]["Close"]
            self.prev_close_price = self.nifty_df.loc[self.idx]["Close"]

        self.num_lots_held = 0
        self.episode += 1

        next_state = self.nifty_df.iloc[self.idx]["State"]
        self.init_csv_file()
        return next_state

    def render(self, mode="human", close=False):
        return None

    def evaluate_model(self):
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
