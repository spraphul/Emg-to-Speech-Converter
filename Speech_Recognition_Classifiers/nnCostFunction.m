function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size_1,...
                                   hidden_layer_size_2, ...
                                   hidden_layer_size_3,...
                                   num_labels, ...
                                   X, y, lambda)

                                   
Theta1 = reshape(nn_params(1:hidden_layer_size_1 * (input_layer_size + 1)), ...
                 hidden_layer_size_1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size_1 * (input_layer_size + 1)))...
                 :(hidden_layer_size_1 * (input_layer_size + 1))+...
                 hidden_layer_size_2 * (hidden_layer_size_1 + 1)), ...
                 hidden_layer_size_2, (hidden_layer_size_1 + 1));
                 
Theta3 = reshape(nn_params((1 + (hidden_layer_size_1 * (input_layer_size + 1)) +...
                 hidden_layer_size_2 * (hidden_layer_size_1 + 1))...
                 :(hidden_layer_size_1 * (input_layer_size + 1))+...
                 hidden_layer_size_2 * (hidden_layer_size_1 + 1)+...
                 hidden_layer_size_3 * (hidden_layer_size_2 + 1)), ...
                 hidden_layer_size_3, (hidden_layer_size_2 + 1));

Theta4 = reshape(nn_params((1 + (hidden_layer_size_1 * (input_layer_size + 1)) +...
                 hidden_layer_size_2 * (hidden_layer_size_1 + 1))+...
                 hidden_layer_size_3 * (hidden_layer_size_2 + 1):end), ...
                 num_labels, (hidden_layer_size_3 + 1));                 

% Setup some useful variables
m = size(X, 1);
X=[ones(size(X,1),1) X];       


J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));


Y_dum=[1:num_labels];
%Y_dum=[1 2 3 4 5 6 7 8]; eight words to classify
Y=Y_dum;
Yprev=(Y_dum==y(1));

for i=2:m
  Y=Y_dum;
  
  Y=(Y==y(i));
  Y=[Yprev;Y];
  Yprev=Y;
  
  
end;


%         Feedforward the neural network and return the cost in the
%         variable J.

predict1=X*Theta1';
predict1=sigmoid(predict1);
predict1=[ones(size(predict1,1),1) predict1];

predict2=predict1*Theta2';
%predict=predict';
predict2=sigmoid(predict2);
predict2=[ones(size(predict2,1),1) predict2];

predict3=predict2*Theta3';
predict3=sigmoid(predict3);
predict3=[ones(size(predict3,1),1) predict3];

predict=predict3*Theta4';
predict=sigmoid(predict);








h=predict;

mat=Y.*log(h)+(1-Y).*log(1-h);

J=(-1/m)*sum(mat(:));

%         Implementing the backpropagation algorithm to compute the gradients

Del1=0;
Del2=0;
Del3=0;
Del4=0;

for t=1:m
  a1=X(t,:)';
  z2=Theta1*a1;
  a2=sigmoid(z2);
  a2=[ones(1,1);a2];
  z3=Theta2*a2;
  a3=sigmoid(z3);
  a3=[ones(1,1);a3];
  z4=Theta3*a3;
  a4=sigmoid(z4);
  a4=[ones(1,1);a4];
  z5=Theta4*a4;
  a5=sigmoid(z5);
  
  h=a5;
  
  delta5=h-Y(t,:)';
  delta4=(Theta4'*delta5).*[ones(1,1);sigmoidGradient(z4)];
  delta4=delta4(2:end,:);
  delta3=(Theta3'*delta4).*[ones(1,1);sigmoidGradient(z3)];
  delta3=delta3(2:end,:);
  delta2=(Theta2'*delta3).*[ones(1,1);sigmoidGradient(z2)];
  delta2=delta2(2:end,:);
  
  Del1=Del1+delta2*a1';
  Del2=Del2+delta3*a2';
  Del3=Del3+delta4*a3';
  Del4=Del4+delta5*a4';
end;


D1=(1/m)*Del1;
D2=(1/m)*Del2;
D3=(1/m)*Del3;
D4=(1/m)*Del4;


Theta1_grad=D1;
Theta2_grad=D2;
Theta3_grad=D3;
Theta4_grad=D4;

% Implementing regularization with the cost function and gradients.  
  
J=J+(lambda/(2*m))*(sum(nn_params.^2)-sum(Theta1(:,1).^2)-sum(Theta2(:,1).^2)...
                                     -sum(Theta3(:,1).^2)-sum(Theta4(:,1).^2));


Theta1_grad=Theta1_grad+[zeros(size(Theta1_grad,1),1) (lambda/m)*Theta1(:,2:end)];
Theta2_grad=Theta2_grad+[zeros(size(Theta2_grad,1),1) (lambda/m)*Theta2(:,2:end)];
Theta3_grad=Theta3_grad+[zeros(size(Theta3_grad,1),1) (lambda/m)*Theta3(:,2:end)];
Theta4_grad=Theta4_grad+[zeros(size(Theta4_grad,1),1) (lambda/m)*Theta4(:,2:end)];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:) ; Theta4_grad(:)];


end
