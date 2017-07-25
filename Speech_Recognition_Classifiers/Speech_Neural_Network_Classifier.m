

%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 32;  
hidden_layer_size_1 = 8;
hidden_layer_size_2 = 8;
hidden_layer_size_3 = 8;   
num_labels = 8;         

%% =========== Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('matlabxy.mat');
m = size(x, 1);


fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size_1);
initial_Theta2 = randInitializeWeights(hidden_layer_size_1, hidden_layer_size_2);
initial_Theta3 = randInitializeWeights(hidden_layer_size_2, hidden_layer_size_3);
initial_Theta4 = randInitializeWeights(hidden_layer_size_3, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; initial_Theta4(:)];


%% =============== Implement Backpropagation ===============

%fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
%checkNNGradients;

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;



%% =================== Training NN ===================


fprintf('\nTraining Neural Network... \n')


options = optimset('MaxIter', 100);


lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size_1,...
                                   hidden_layer_size_2, ...
                                   hidden_layer_size_3,...
                                   num_labels, x, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta's back from nn_params
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

fprintf('Program paused. Press enter to continue.\n');
pause;




%% ================= Implement Predict =================
 
pred = predict(Theta1, Theta2, Theta3, Theta4, x);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


