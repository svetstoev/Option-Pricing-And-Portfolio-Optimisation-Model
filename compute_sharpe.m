function [ result ] = compute_sharpe( weights, ret_is, cov_mat_is )

% weights are suppose to be a column vector
pvar = weights' * cov_mat_is * weights;
pret = weights' * ret_is;

% assumes rf = 3% annually => (3/252) daily
result = (pret - 0.03/252) / sqrt(pvar); 

end
