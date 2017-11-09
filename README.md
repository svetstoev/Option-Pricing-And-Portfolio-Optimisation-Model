# Portfolio-Optimisation-and-Option-Pricing-Model

The following project serves two purposes. First, the program constructs an optimal portfolio in a universe of 30 US stocks and backtests the performance of several portfolio strategies using historical data. Second,using Black-Scholes Model, the project builds an option simulator.

**1. Initial requirements**

In order to make use of the project, the user should have a Matlab- version R2016b installed on their PC, so that they can compile and run the codes included in this repository.

**2. Getting started**

In order to run the model the user just needs to open open and run the file called `main.m`

**3. Repository constituents**

- `main.m` - As the name suggests, this is our main file which assembles the functionalities of the other 3
- `BlackScholesPrice.m` - Computes the price of an option using the BS formula and taking as inputs the stock and strike prices, time to expiration, interest rate, volatility and also information of whether the option is a call or a put (i.e. `S, K, T, r, sigma, CallorPut`)
- `compute_pvar.m` - Computes the portfolio variance taking as inputs the portfolio weights and the covariance matrix (i.e. `weights, cov_mat_is`)
- `compute_sharpe.m` - Computes the Sharpe Ratio using as inputs the portfolio weights, return matrix and covariance matrix (i.e. `weights, ret_is, cov_mat_is`)
- `README.md` - the file that you are currently reading

**5. License**

The files included in this repository are covered under the MIT license.

**6. Authors**

Svetlozar Stoev
