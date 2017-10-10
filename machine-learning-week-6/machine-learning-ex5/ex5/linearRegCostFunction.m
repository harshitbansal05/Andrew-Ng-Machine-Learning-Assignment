function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%X = [ones(m,1) X];

diff = (X * theta - y).^2;
J = sum(diff)/(2*m);
reg_J = 0;
reg_grad = zeros(size(theta, 1), 1);
for i = 2:length(theta)
    reg_J = reg_J + theta(i).^2;
    reg_grad(i) = theta(i); 
end
J = J + lambda * reg_J/(2*m);

for i = 1:m
    grad = grad + (X(i,:) * theta - y(i)) * X(i,:).';
end
grad = grad/m + lambda * reg_grad/m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
