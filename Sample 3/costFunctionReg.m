function [J, grad] = costFunctionReg(theta, X, y, lambda)
% COSTFUNCTIONREG Compute cost and gradient for logistic regression with
% regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost with respect to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the 
%               partial derivatives of the cost with respect to each
%               parameter in theta.

theta_reg=[0; theta(2:end, :)];
J = (1/m)*(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))...
      +(1/2*lambda/m)*(theta_reg'*theta_reg);
grad = (1/m)*(X'*(sigmoid(X*theta)-y) + lambda * theta_reg);


% =============================================================

end
