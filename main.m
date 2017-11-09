%% Question 1

% 1.1

T=readtable('equity_dataset_30.csv'); %read the data from the 'equity_dataset_30.csv'

%create a matrix including numerical data only
px.mat=table2array(T(1:2641,2:31)); %for the column dimension our start value is '2' since we are excluding the 'Date' column

%1.2

% Calculate the returns for all the stocks for each date
ret_mat = diff(log(px.mat));

%1.3

Dates=datenum(T{:,1}); %Convert all the dates from a string to a number format 

% Convert the start and end dates for both samples to a number format
start_date_is = datenum('2005-01-03');
end_date_is = datenum('2012-12-31');
start_date_oos = datenum('2013-01-02'); %choose a cut-off date
end_date_oos = datenum('2015-06-30');

% Find the indeces of the dates above so that we can work easily with our
% data
start_date_is_index = find(Dates == start_date_is)
end_date_is_index = find(Dates == end_date_is)
start_date_oos_index = find(Dates == start_date_oos)
end_date_oos_index = find(Dates == end_date_oos)

%is_index = Dates < cut_off;
%os_index = ~is_index;

% Split our dates into two sets - In-Sample dates and Out-of-Sample dates
% (a column vector)
Dates_is = Dates(start_date_is_index:end_date_is_index,1);
Dates_oos = Dates(start_date_oos_index:end_date_oos_index,1);

%get the prices for the two sample periods (training and testing)
%price_is = px.mat(is_index,:); 
%price_oos = px.mat(~is_index,:);

% Calculate the returns for the stocks for the whole 8-year period using
% the 'px.mat' from above
ret_mat_total=diff(log(px.mat));

% Now that we have the start and date indeces it is easy to set the intervals for our two samples-training and testing.
ret_mat_is= ret_mat_total(start_date_is_index:end_date_is_index-1,:); %we need to subtract '1' from the end price in order to exclude the return for '2013-01-02'
ret_mat_oos= ret_mat_total(start_date_oos_index-1:end_date_oos_index-1,:); %we subtract the '1' from the start here in order to include the return for '2013-01-2' which should be in the Out-of-sample; we also need to subtract '1' in the end in order to comensate for the fact that vectors are of different lengths

%ret_mat_is=diff(log(price_is));
%ret_mat_oos=diff(log(price_oos));

%1.4

%take the average returns for the 30 stocks over the 8-year period
histret_mat_is=mean(ret_mat_is,1); %the '1' stands for the dimension (although the function would work even if we omit it)
cov_mat_is=cov(ret_mat_is); %create a covariance matrix whose main diagonal represents the variances of the returns 

%1.5

%assume annual risk-free rate is 0.03, which means that the daily risk-free
%rate can be found as 0.03/252 - assuming 252 trading days per year
ret_is=histret_mat_is' %transpose to convert to a column vector
N=length(ret_is) %number of stocks

Aeq = ones(1,N); %create a row vector with each element equal to '1' and a length equal to the number of stocks
beq = 1; %a scalar
%w0 = ones(N,1)*(1/N); 

%1.5.1: Equally weighted portfolio (Benchmark portfolio)

w0 = ones(N,1)*(1/N); %the weight assigned to each of the stocks in the benchmark portfolio
%portfolio0_return=sum((1/N)*histret_mat_is)
sr0=compute_sharpe(w0,ret_is,cov_mat_is)

%1.5.2: Maximize sharpe (short-selling allowed) => w1 can be negative

options = optimset('Display', 'off'); % suppress optimization message

w1 = fmincon(@(w)-compute_sharpe(w, ret_is, cov_mat_is), w0, [], [], Aeq, beq, [], [], [], options); %calculate the weights for each of the stocks
sr1 = compute_sharpe(w1, ret_is, cov_mat_is); %calculate the SR using the function compute_sharpe()
fprintf('The maximized Sharpe Ratio is %.4f.\n', sr1);

%1.5.3: Maximize sharpe (no short-selling)

% w2>=0, the weights cannot be negative
lb=zeros(1,N);

w2 = fmincon(@(w)-compute_sharpe(w, ret_is, cov_mat_is), w0, [], [], Aeq, beq, lb, [], [], options); %calculate the weights for each of the stocks
sr2 = compute_sharpe(w2, ret_is, cov_mat_is); %calculate the SR using the function compute_sharpe()
fprintf('The maximized Sharpe Ratio is %.4f.\n', sr2);

%1.5.4: Minimize portfolio variance (short-selling allowed)

% w3 can be negative

w3 = fmincon(@(w)compute_pvar(w, cov_mat_is), w0, [], [], Aeq, beq, [], [], [], options); %calculate the weights for each of the stocks
pvar3 = compute_pvar(w3, cov_mat_is); %calculate the variance of the MVP using the function compute_pvar()
fprintf('The variance of constrainted MVP is %.4f.\n', pvar3);

%1.6: In-Sample (IS)

%Step 1: Calculate the daily returns for the four portfolios

ret_daily_isp0 = ret_mat_is*w0;
ret_daily_isp1 = ret_mat_is*w1;
ret_daily_isp2 = ret_mat_is*w2;
ret_daily_isp3 = ret_mat_is*w3;

%Step 2: Construct the equity curves using the IS daily returns calculated
%in step 1

equity_curve_isp0 = cumprod(1+ret_daily_isp0)
equity_curve_isp1 = cumprod(1+ret_daily_isp1)
equity_curve_isp2 = cumprod(1+ret_daily_isp2)
equity_curve_isp3 = cumprod(1+ret_daily_isp3)

%Step 3: Plot the equity curves for the Benchmark and Portfolio 1

clf;
figure(1);
plot(Dates_is(2:end), log10(equity_curve_isp0)) % Benchmark Portfolio using ln/ we start from date 2 because the return matrix starts one date later than the first date in the set
datetick('x')

hold on % we use 'hold on' so that we can plot the two equity curves on one graph
plot(Dates_is(2:end), log10(equity_curve_isp1)) % Benchmark Portfolio
datetick('x')
%ylim('auto');
xlabel('Date');
ylabel('Return');
legend('Equity Curve for the Benchmark Portfolio','Equity Curve for Portfolio 2');
legend('Location','Northwest');
title('Equity Curves for the Benchmanrk and Portfolio 2 (Using In-sample data)');

%Step 4: Calculate the annualized returns and SRs for each strategy

% Equally weighted (benchmark) portfolio:
ret_annual_isp0 = mean(ret_daily_isp0) * 252;
std_annual_isp0 = std(ret_daily_isp0) * sqrt(252); 
sharpe_ratio_annual_isp0 = (ret_annual_isp0 - 0.03) / std_annual_isp0;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_isp0);

% Portfolio 1:
ret_annual_isp1 = mean(ret_daily_isp1) * 252;
std_annual_isp1 = std(ret_daily_isp1) * sqrt(252); 
sharpe_ratio_annual_isp1 = (ret_annual_isp1 - 0.03) / std_annual_isp1;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_isp1);

%Portfolio 2:
ret_annual_isp2 = mean(ret_daily_isp2) * 252;
std_annual_isp2 = std(ret_daily_isp2) * sqrt(252); 
sharpe_ratio_annual_isp2 = (ret_annual_isp2 - 0.03) / std_annual_isp2;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_isp2);

% Portfolio 3:
ret_annual_isp3 = mean(ret_daily_isp3) * 252;
std_annual_isp3 = std(ret_daily_isp3) * sqrt(252); 
sharpe_ratio_annual_isp3 = (ret_annual_isp3 - 0.03) / std_annual_isp3;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_isp3);

%Step 5: Calculate the Cumulative Average Return (CAR) for each strategy
% our In-sample period is from '2005-01-03' to '2012-12-31', i.e. 8 years

CAR_isp0 = (equity_curve_isp0(end)/equity_curve_isp0(1))^(1/8) - 1; fprintf('Average Annual Return is %.4f \n', CAR_isp0);
CAR_isp1 = (equity_curve_isp1(end)/equity_curve_isp1(1))^(1/8) - 1; fprintf('Average Annual Return is %.4f \n', CAR_isp1);
CAR_isp2 = (equity_curve_isp2(end)/equity_curve_isp2(1))^(1/8) - 1; fprintf('Average Annual Return is %.4f \n', CAR_isp2);
CAR_isp3 = (equity_curve_isp3(end)/equity_curve_isp3(1))^(1/8) - 1; fprintf('Average Annual Return is %.4f \n', CAR_isp3);

%1.7: Out-of-Sample (OOS)

%Step 1: Calculate the daily returns for the four portfolios

ret_daily_oosp0 = ret_mat_oos*w0;
ret_daily_oosp1 = ret_mat_oos*w1;
ret_daily_oosp2 = ret_mat_oos*w2;
ret_daily_oosp3 = ret_mat_oos*w3;

%Step 2: Construct the equity curves using the OOS daily returns calculated
%above

equity_curve_oosp0 = cumprod(1+ret_daily_oosp0)
equity_curve_oosp1 = cumprod(1+ret_daily_oosp1)
equity_curve_oosp2 = cumprod(1+ret_daily_oosp2)
equity_curve_oosp3 = cumprod(1+ret_daily_oosp3)

%Step 3: Plot the equity curves for the Benchmark and Portfolio 1


figure(2);
plot(Dates_oos, log10(equity_curve_oosp0)) % Benchmark Portfolio using log10
datetick('x')
hold on % we use 'hold on' so that we can plot the two equity curves on one graph
plot(Dates_oos, log10(equity_curve_oosp1)) % Benchmark Portfolio
datetick('x')
xlabel('Date');
ylabel('Return');
legend('Equity Curve for the Benchmark Portfolio','Equity Curves for the Benchmanrk and Portfolio 2');
legend('Location','Northwest');
title('Equity Curves for the Benchmanrk and Portfolio 2 (Using Out-of-sample data)');

%Step 4: Calculate the annualized returns and SRs for each strategy

% Equally weighted (benchmark) portfolio:
ret_annual_oosp0 = mean(ret_daily_oosp0) * 252;
std_annual_oosp0 = std(ret_daily_oosp0) * sqrt(252); 
sharpe_ratio_annual_oosp0 = (ret_annual_oosp0 - 0.03) / std_annual_oosp0;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_oosp0);

% Portfolio 1:
ret_annual_oosp1 = mean(ret_daily_oosp1) * 252;
std_annual_oosp1 = std(ret_daily_oosp1) * sqrt(252); 
sharpe_ratio_annual_oosp1 = (ret_annual_oosp1 - 0.03) / std_annual_oosp1;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_oosp1);

%Portfolio 2:
ret_annual_oosp2 = mean(ret_daily_oosp2) * 252;
std_annual_oosp2 = std(ret_daily_oosp2) * sqrt(252); 
sharpe_ratio_annual_oosp2 = (ret_annual_oosp2 - 0.03) / std_annual_oosp2;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_oosp2);

% Portfolio 3:
ret_annual_oosp3 = mean(ret_daily_oosp3) * 252;
std_annual_oosp3 = std(ret_daily_oosp3) * sqrt(252); 
sharpe_ratio_annual_oosp3 = (ret_annual_oosp3 - 0.03) / std_annual_oosp3;
fprintf('Sharpe Ratio is %.2f \n', sharpe_ratio_annual_oosp3);

%Step 5: Calculate the Cumulative Average Return (CAR) for each strategy
% our Out-of-sample period is from '2013-01-02' to '2015-06-30', i.e. 2.5 years

CAR_oosp0 = (equity_curve_oosp0(end)/equity_curve_oosp0(1))^(1/2.5) - 1; fprintf('Average Annual Return is %.4f \n', CAR_oosp0);
CAR_oosp1 = (equity_curve_oosp1(end)/equity_curve_oosp1(1))^(1/2.5) - 1; fprintf('Average Annual Return is %.4f \n', CAR_oosp1);
CAR_oosp2 = (equity_curve_oosp2(end)/equity_curve_oosp2(1))^(1/2.5) - 1; fprintf('Average Annual Return is %.4f \n', CAR_oosp2);
CAR_oosp3 = (equity_curve_oosp3(end)/equity_curve_oosp3(1))^(1/2.5) - 1; fprintf('Average Annual Return is %.4f \n', CAR_oosp3);

%% Comments

% The respective In-Sample Sharpe ratios  for the Benchmark Portfolio, 
% Portfolio 1, Portfolio2 and Portfolio 3 are {0.01, 1.68, 0.74, 0.59}

% Analogically, the respective Out-of-Sample Sharpe ratios for the 
% four portfolios are {1.12, -0.35, 1.05, 0.40}

% You can find further discussion of the results in the report

%% Question 2

2.1

clear all
close all
clc

% dt = 1/252 yr, T = 1 yr, time gride = T/dt + 1 = 253 grid points
dt = 1/252;
T = 1;
tgrid = 0: dt: T;
N = length(tgrid);

% Set up parameters & initialize the price vector
S0 = 100;
mu = 0.15;
sigma = 0.3;
S = zeros(1,N);

% Simulate random number epsilon 
eps = randn(1,N);
% Simulate stock prices
S_T = S0*exp((mu-0.5*(sigma^2))*tgrid+sigma*sqrt(tgrid).*eps)
plot(tgrid, S_T)
%legend('Stock price')
title('Simulated path of a stock price')
xlabel('Time(yr)')
ylabel('Asset Price($)')

2.2

clear all;
clc;

S0 = 100; % Value of the underlying
K = 100; % Strike (exercise price)
T = 1;   % Maturity
mu = 0.15; % Stock price mean
r = 0.10; % Risk free interest rate
sigma = 0.30; % Volatility
% Monte-Carlo Method Parameters 
M=10000000; % Number of Monte-Carlo trials
eps = randn(M,1);
S_T = S0*exp((mu-0.5*(sigma^2))*T+sigma*sqrt(T).*eps);
option_call=max(S_T-K,0); % Evaluate the Put option options
option_put=max(K-S_T,0); % Evaluate the Put option options
option_values = [option_call option_put];
present_vals=exp(-r*T)*option_values; % Discount under r-n assumption

options=mean(present_vals); % Take the average
display('call_value, put_value')
display(options)

% 2.3: BlackScholes

S = 100; % Value of the underlying
K = 100; % Strike (exercise price)
T = 1;   % Maturity
r = 0.10; % Risk free interest rate
sigma = 0.30; % Volatility
% parameters = [S, K, T, r, sigma];
% call functions to calculate the price of the option 
CallPrice = BlackScholesPrice(S, K, T, r, sigma, 'Call');
PutPrice = BlackScholesPrice(S, K, T, r, sigma, 'Put');
% fprintf output 
fprintf('\n The Price of a European Call is :     %8.2f        \n' , CallPrice);
fprintf('\n The Price of a European Put is :      %8.2f        \n'  , PutPrice);

%2.4

%% 
% Call (BS) = $16.73
% Put (BS) = $7.22

% Call (MC) = $20.40
% Put (MC) = $5.76

%Comparing the resutls from the Monte Carlo (MC) simulation and Black-Sholes (BS), we see that
%the call price is lower and the put price is higher under BS assumptions(see above).  

