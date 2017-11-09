function price = BlackScholesPrice(S, K, T, r, sigma, CallorPut)
% This function calculates option price base on the Black-Schole formula.

% Input: S: spot stock price
%        K: strike price
%        T: maturity
%        r: interest rate
%        sigma: volatility
%        callorput: user-defined string input as 'Call' or 'Put' option

if strcmp(CallorPut,'Call') == 1
    phi = 1;
elseif strcmp(CallorPut,'Put') == 1
    phi = -1;
else
    error('Invalid Option Type')
end

d1 = (log(S/K) + (r + 0.5 * sigma^2)* T)./ sigma.* sqrt(T);
Nd1 = normcdf(phi*d1,0,1); 

d2 = d1 - sigma.* sqrt(T);
Nd2 = normcdf(phi*d2,0,1); 

price = phi.*S.*Nd1 - phi.*K.*exp(-r * T).*Nd2;
end
