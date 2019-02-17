# Python VaR

This is a simple implementations of the VaR (Value at Risk) model. The VaR is a simple model to evaluate the risk of your portfolio.
The VaR is a single number that indicates the likehood of losing X percent of capital given a confidence interval and at any given trading session. 

A VaR of -4% with a confidence interval of 5% should be understoo as the following:
  
+ *"There is a 5% chance that I will lose 4% or more of my portfolio in any given trading day"*

The calculation of the VaR requires the calculation of the portfolio's standard deviations as a pre-requisite. The standard deviation
of a portfolio with more than one asset can be calculated using the matricial implementation of markowitz's formula, which can be found [here](https://en.wikipedia.org/wiki/Modern_portfolio_theory "Modern Portfolio Theory")

In the real world, the biggest difficulty in the implementation of this kind of project is the persistency of the data sources (both weights and historical prices). Sometimes the weights are not that simple to calculate because one fund may have trade in many different accounts and sometimes historical prices are difficult to get because of iliquid assets (like iliquid bonds and iliquid options) and how NAs will be dealt must be discussed. In the case of iliquid options, the Black-Scholes pricing model can be applied to generate historical data.
