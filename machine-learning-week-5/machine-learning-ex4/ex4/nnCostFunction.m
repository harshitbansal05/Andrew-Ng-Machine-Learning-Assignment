function [J grad] = nnCostFunction(nn_params, ...
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
X = [ones(m,1) X];
pred = (Theta2 * [ones(m,1) sigmoid(Theta1 * X.').'].').';
a = zeros(m ,10);
for i = 1:length(y)
    a(i, y(i)) = 1;
end
error = 0;
for i = 1:m
    found = sigmoid(pred(i,:));
    for j = 1:num_labels
        error = error - a(i,j) * log(found(1,j)) - (1 - a(i,j)) * log(1 - found(1,j));
    end
end
error = error/m;
J = error;

   reg_J = 0; 
   reg_grad_1 = zeros(hidden_layer_size, input_layer_size + 1);
   reg_grad_2 = zeros(num_labels, hidden_layer_size + 1);
   for i = 1:hidden_layer_size
        for j = 2:input_layer_size + 1
            reg_J = reg_J + Theta1(i,j) * Theta1(i,j);
            reg_grad_1(i,j) = reg_grad_1(i,j) + Theta1(i,j);
        end
    end
    for i = 1:num_labels
        for j = 2:hidden_layer_size + 1
            reg_J = reg_J + Theta2(i,j) * Theta2(i,j);
            reg_grad_2(i,j) = reg_grad_2(i,j) + Theta2(i,j);
        end
    end
    J = J + lambda * reg_J/(2*m); 
    
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
grad_1 = zeros(hidden_layer_size, input_layer_size + 1);
grad_2 = zeros(num_labels, hidden_layer_size + 1);

for t = 1:m
    a_1 = sigmoid(Theta1 * X(t,:).');
    a_1 = [1;a_1];
    %disp(size(a_1));
    a_2 = sigmoid(Theta2 * a_1);
    b = zeros(num_labels, 1);
    b(y(t),1) = 1;
    found = a_2;
    del3 = found - b;
    del2 = (Theta2.' * del3) .* a_1 .* (ones(hidden_layer_size + 1,1) - a_1);
    del2 = del2(2:end);
    grad_1 = grad_1 + del2 * X(t,:);
    grad_2 = grad_2 + del3 * a_1.';
end    
Theta1_grad = grad_1/m + lambda * reg_grad_1/m;
Theta2_grad = grad_2/m + lambda * reg_grad_2/m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
