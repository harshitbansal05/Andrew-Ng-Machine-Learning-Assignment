function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
total = 0;
for i = 1:m
    pred = sigmoid(X(i,:)*theta);
    cost = -y(i)*log(pred) - (1-y(i))*log(1-pred);
	total = total + cost;
	for j = 1:length(theta)
		grad(j) = grad(j) + (pred - y(i))*X(i, j);	
	end
end	
for j = 2:length(theta)
	
	grad(j) = grad(j) + lambda*theta(j);
	regCost = (lambda/2)*theta(j).^2;
	total = total + regCost;
end	
% =============================================================
J = total/m;
grad = grad/m;
end
