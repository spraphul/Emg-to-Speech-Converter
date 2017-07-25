%==========Softmax Classifier For Speech Recognition ========================


clear ; close all; clc


inputSize = 32; 
numClasses = 8;     

lambda = 1e-4; % Weight decay parameter
%lambda = 0.1;
%%======================================================================
%% STEP 1: Load data
fprintf('Loading and Visualizing Data ...\n')
load('matlab.mat');
%m = size(x, 1);
x=x';
y=y';
%inputData = images;
m = size(x, 2);
sel=randperm(m);
sel1=sel(1:m-100);

xtrain=x(:,sel1);
ytrain=y(:,sel1);


fprintf('Program paused. Press enter to continue.\n');
pause;

%%======================================================================
%% STEP 4: Learning parameters
%
 
%  you can start training your softmax regression code using softmaxTrain
%  (which uses minFunc).

options.maxIter = 1000;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            xtrain, ytrain, options);
                          


%%======================================================================
%% STEP 5: Testing

[pred] = softmaxPredict(softmaxModel, xtrain);

acc = mean(ytrain(:) == pred(:));
fprintf('Train Accuracy: %0.3f%%\n', acc * 100);


%load('matlabxy.mat');
sel2=sel(m-100+1:end);
xtest=x(:,sel2);
ytest=y(:,sel2);


[pred] = softmaxPredict(softmaxModel, xtest);

acc = mean(ytest(:) == pred(:));
fprintf('Test Accuracy: %0.3f%%\n', acc * 100);


