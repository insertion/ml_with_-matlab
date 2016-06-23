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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h =X*theta;

J =trace((h-y)'*(h-y))/m/2 + lambda/2/m*theta(2:end)'*theta(2:end);
grad(1)=(h-y)'*X(:,1)/m;
%����Ҫע��theta�����Ŀ���չ��
for i = 2:numel(theta)
    grad(i)=(h-y)'*X(:,i)/m+lambda/m*theta(i);
end











% =========================================================================

grad = grad(:);

end
