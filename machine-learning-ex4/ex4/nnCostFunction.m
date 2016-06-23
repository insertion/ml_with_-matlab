function [J,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ ones(m,1) X];         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
h =  sigmoid([ones(m,1) sigmoid(X*Theta1')]*Theta2');%m*k
e3 = zeros(size(h));
for i=1:num_labels
    yk = (y==i);
    e3(:,i)= h(:,i)-yk;%5000个样本，十个类别的误差 5000*10
    J = J -  log(h(:,i))'*yk - (1-yk)'*log(1-h(:,i));
end
e2 = e3 *Theta2(:,2:end).*sigmoidGradient(X*Theta1');%X*Theta1'=z(2) 5000*25

Theta1_grad  =(e2'* X)/m; %25 * 401
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda*Theta1(:,2:end)/m;
Theta2_grad  =(e3'* [ones(m,1) sigmoid(X*Theta1')])/m;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda*Theta2(:,2:end)/m;
%梯度的个数和参数theta的个数一样
r =trace(Theta1(:,2:end)'*Theta1(:,2:end)) ...
    + trace(Theta2(:,2:end)'*Theta2(:,2:end));
    %sum(sum(Theta1(:,2:end).*Theta1(:.2:end)))
    %不能reg 第0项
J = J/m +lambda*r/(2*m);
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
