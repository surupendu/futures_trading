from math import sqrt

class Evaluate:
    '''
        Evaluate the trading models
    '''
    def __init__(self, trade_df, balance):
        self.trade_df = self.eod(trade_df)
        self.initial_balance = balance
    
    def eod(self, trade_df):
        trade_df["Date_1"] = trade_df["Date"].shift(-1)
        trade_df["EOD"] = (trade_df["Date_1"] != trade_df["Date"])
        trade_df.drop(columns=["Date_1"], inplace=True)
        return trade_df
    
    def day_wise_returns(self, trade_df):
        '''
            Calculate the return at the end of the trading day
        '''
        trade_df = trade_df[trade_df["EOD"] == True]
        trade_df["Balance_1"] = trade_df["Balance"].shift(1)
        trade_df["Balance_1"] = trade_df["Balance_1"].fillna(value=self.initial_balance)
        trade_df["Return"] = (trade_df["Balance"] - trade_df["Balance_1"])/trade_df["Balance_1"]
        returns = trade_df["Return"]
        return returns

    # Calculate the Sharpe Ratio
    def calc_sharpe_ratio(self, risk_free_rate=None, annualized_coefficient=None):
        returns = self.day_wise_returns(self.trade_df)
        sharpe_ratio = returns.mean()/returns.std()
        return sharpe_ratio

    # Calculate the Sortino Ratio
    def calc_sortino_ratio(self, risk_free_rate=None, annualized_coefficient=None):
        returns = self.day_wise_returns(self.trade_df)
        neg_returns = returns[returns<0]
        sortino_ratio = returns.mean()/neg_returns.std()
        return sortino_ratio

    # Calculate the total profit at the end of the trading session
    def calc_profit(self):
        profit  = self.trade_df.iloc[-1]["Balance"] - self.initial_balance
        return profit

    # Calculate MDD percentage and duration
    def calc_max_drawdown(self):
        returns = self.day_wise_returns(self.trade_df)
        cumulative_returns = (returns + 1).cumprod()
        peak_returns = cumulative_returns.expanding(min_periods=1).max()
        draw_downs = (cumulative_returns/peak_returns) - 1
        max_draw_down = draw_downs.min() * 100
        draw_down_duration = draw_downs.argmin()
        return max_draw_down, draw_down_duration
    
    # Calculate return percentage at the end of trading session
    def calc_annualized_return(self):
        final_balance = self.trade_df.iloc[-1]["Balance"]
        annualized_return = ((final_balance - self.initial_balance)/self.initial_balance) * 100
        return annualized_return
    
    # Calculate volatility in the trading session
    def calc_annualized_volatility(self):
        returns = self.day_wise_returns(self.trade_df)
        volatility = returns.std() * sqrt(len(returns))
        return volatility
