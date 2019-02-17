from scipy import stats as st
from dateutil import parser
import pandas as pd
import numpy as np
import requests
import json

class simple_var_model(object):
    
    def __init__(self):
        self.alpha_api_key = "WQQ3DSE4T5GCB7TW"
        self.stocks = ["AAPL", "MSFT", "AMZN"]
        self.confidence = 0.95
        self.number_of_days = 62
        self.weights = None

    def set_dummy_weights(self):
        # Sets dummy weights for the portfolios
        # Main assumption here is that all assets have same weights
        self.weights = 1.0 / len(self.stocks)
        self.weights = np.repeat(self.weights, len(self.stocks), axis=0)
        return self.weights

    def get_data(self, stock):
        # Downloads data from alphavantage
        query = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" +  stock + "&apikey="+ self.alpha_api_key
        r = requests.get(query)
        return r.text

    def create_df(self, return_type="close"):
        """
        Other options for retrun_type: open, high, low and volume.
        If return type is equal none, it will return all options above        
        """
        # Creates empty DF
        df_final = pd.DataFrame()
        # Loops over each stock
        for stock in self.stocks:
            # Retrives get data
            data = json.loads(self.get_data(stock))
            # Loads data in a DataFrame
            df = pd.DataFrame(data["Time Series (Daily)"]).T        
            # Renames columns
            df.columns = [stock + name[2:] for name in df.columns]
            # Concants DFs
            df_final = pd.concat([df, df_final], axis=1)
        # Returns close by defaul
        if return_type:
            df_final = df_final.loc[:, df_final.columns.str.contains(return_type)]
        
        # Tranform index to datetime
        df_final.index = pd.to_datetime(df_final.index)
        # Sort by date
        df_final.sort_index(ascending=True, inplace=True)
        # Ttransforms data to float
        self.df_final = df_final.astype(float)
    
    def _subset_data(self):
        """
        In the var model, only the 62 most recent days are taken
        into account for the calculations
        """
        df = self.df_final
        df = df.tail(self.number_of_days)
        return df
    
    def markowitz_formula(self):
        """
        Used to find the standard deviations of the porfolio.
        """
        # Gets data
        df = self._subset_data().pct_change()
        # Calculates covariance matrix
        df_cov = df.cov()
        # Calculates the sigama**2
        sigma_squared = np.dot(self.weights, np.dot(self.weights, df_cov))
        # Calculates sigma
        sigma = np.sqrt(sigma_squared)
        return sigma

    def avg_return(self):
        # Get data
        df = self._subset_data()
        # Calculate pct change
        df = df.pct_change()
        # Calculate the df daily return
        df_daily_return = self.weights.reshape(1, len(self.stocks)) * df
        df_daily_return = df_daily_return.sum(axis=1)
        
        return df_daily_return

    def parametric_var(self):
        # Get how many standard deviations on the bell curve
        bell_std_dev = st.norm.ppf(self.confidence)
        # Average return is the expected return according to the theory.
        avg_return = self.avg_return().mean()
        # Get standard deviation
        sigma = self.markowitz_formula()
        # Compute VaR
        var = avg_return - bell_std_dev * sigma

        return var

    def non_par_var(self):
        df = self.avg_return()
        # Compute non-parametric VaR
        quantile = df.quantile(1.0 - self.confidence)

        return quantile

    def main(self):    
        self.set_dummy_weights()
        self.create_df()
        self.parametric_var()
        self.non_par_var()
        
       
if __name__ == "__main__":
    model = simple_var_model()
    model.main()