function [ result ] = compute_pvar( weights, cov_mat_is )

% weights are suppose to be a column vector
result = weights' * cov_mat_is * weights;

end

