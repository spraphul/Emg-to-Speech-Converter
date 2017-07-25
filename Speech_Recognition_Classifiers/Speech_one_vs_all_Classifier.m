
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 32;  
num_labels = 8;          

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('matlab.mat'); % training data stored in arrays x, y

m = size(x, 1);
sel=randperm(m);
sel1=sel(1:m-100);

xtrain=x(sel1,:);
ytrain=y(sel1,:);



fprintf('Program paused. Press enter to continue.\n');
pause;

%{
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(xtrain, ytrain, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;
%}

%% ================ setting x_train & y_train ================



% Saving trained parameters.....
%save('trained_parameters.mat','all_theta');

sel2=sel(m-100+1:end);
xtest=x(sel2,:);
ytest=y(sel2,:);




%% ================ Plotting optimal lambda ================
%{
[lambda_vec, error_train, error_val] = ...
    validationCurve(xtrain, ytrain, xtest, ytest,num_labels);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Accuracy');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
%}
%% ================ Part 3: Predict for One-Vs-All ================
%
load('trained_parameters.mat');
all_theta=all_theta;

pred = predictOneVsAll(all_theta, xtrain);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytrain)) * 100);


pred = predictOneVsAll(all_theta, xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

