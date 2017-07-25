function [J grad] = dum_nncostfunction(nn_params, ...
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
X=[ones(size(X,1),1) X];       
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Y_dum=[1:num_labels];
%Y_dum=[1 2 3 4 5 6 7 8 9 10];
Y=Y_dum;
Yprev=(Y_dum==y(1));

for i=2:m
  Y=Y_dum;
  
  Y=(Y==y(i));
  Y=[Yprev;Y];
  Yprev=Y;
  
  
end;


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
predict1=X*Theta1';
predict1=sigmoid(predict1);
predict1=[ones(size(predict1,1),1) predict1];

predict=predict1*Theta2';
%predict=predict';
predict=sigmoid(predict);

h=predict;

mat=Y.*log(h)+(1-Y).*log(1-h);

J=(-1/m)*sum(mat(:));
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
Del1=0;
Del2=0;
for t=1:m
  a1=X(t,:)';
  z2=Theta1*a1;
  a2=sigmoid(z2);
  a2=[ones(1,1);a2];
  z3=Theta2*a2;
  a3=sigmoid(z3);
  h=a3;
  
  delta3=h-Y(t,:)';
  delta2=(Theta2'*delta3).*[ones(1,1);sigmoidGradient(z2)];
  delta2=delta2(2:end,:);
  
  Del1=Del1+delta2*a1';
  Del2=Del2+delta3*a2';
end;


D1=(1/m)*Del1;
D2=(1/m)*Del2;

Theta1_grad=D1;
Theta2_grad=D2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

  
  
  
J=J+(lambda/(2*m))*(sum(nn_params.^2)-sum(Theta1(:,1).^2)-sum(Theta2(:,1).^2));


Theta1_grad=Theta1_grad+[zeros(size(Theta1_grad,1),1) (lambda/m)*Theta1(:,2:end)];
Theta2_grad=Theta2_grad+[zeros(size(Theta2_grad,1),1) (lambda/m)*Theta2(:,2:end)];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
