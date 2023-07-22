import pandas as pd
import numpy as np
from talib import RSI
import calendar
import ast
import tqdm as tq

def train_test_split(path, file_name, train_years, test_years):
    nifty_df = pd.read_csv(path + file_name)
    nifty_df["State"] = RSI(nifty_df["Close"], 14)
    nifty_df.dropna(inplace=True)
    # nifty_df["EOC"] = True
    nifty_df["EOC"] = False
    nifty_df = eoc_calc(nifty_df, train_years + test_years)
    
    train_df = pd.DataFrame([])
    test_df = pd.DataFrame([])
    
    for train_year in train_years:
        train_df = train_df.append(nifty_df[nifty_df["Date"].str.contains(train_year)])
    train_df["State"] = train_df["State"].apply(lambda x: int(np.floor(x)))
    train_df.reset_index(drop=True, inplace=True)

    for test_year in test_years:
        test_df = test_df.append(nifty_df[nifty_df["Date"].str.contains(test_year)])
    test_df["State"] = test_df["State"].apply(lambda x: int(np.floor(x)))
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df

def eoc_calc(nifty_df, years):
    for year in tq.tqdm(years):
        months = [i+1 for i in range(12)]
        year = ast.literal_eval(year)
        for month in months:
            weeks = calendar.monthcalendar(year, month)
            if weeks[-1][calendar.THURSDAY]:
                day = weeks[-1][calendar.THURSDAY]
                if month <= 9:
                    date = "{:}-0{:}-{:}".format(year, month, day)
                elif month > 9:
                    date = "{:}-{:}-{:}".format(year, month, day)
            elif weeks[-2][calendar.THURSDAY]:
                day = weeks[-2][calendar.THURSDAY]
                if month <= 9:
                    date = "{:}-0{:}-{:}".format(year, month, day)
                elif month > 9:
                    date = "{:}-{:}-{:}".format(year, month, day)

            if len(nifty_df[nifty_df["Date"].str.contains(date)]) > 0:
                time = nifty_df[nifty_df["Date"].str.contains(date)].iloc[-1]["Time"]
                nifty_df.loc[(nifty_df["Date"]==date) & (nifty_df["Time"]==time), "EOC"] = True
            else:
                while len(nifty_df[nifty_df["Date"].str.contains(date)]) == 0:
                    day = day - 1
                    if month <= 9:
                        date = "{:}-0{:}-{:}".format(year, month, day)
                    elif month > 9:
                        date = "{:}-{:}-{:}".format(year, month, day)
                time = nifty_df[nifty_df["Date"].str.contains(date)].iloc[-1]["Time"]
                nifty_df.loc[(nifty_df["Date"]==date) & (nifty_df["Time"]==time), "EOC"] = True
    return nifty_df
