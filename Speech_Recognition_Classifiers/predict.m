function p = predict(Theta1, Theta2, Theta3, Theta4, X)
% PREDICT Predict the label of an input given a trained neural network
% words in our case

% Useful values
m = size(X, 1);
num_labels = size(Theta4, 1);

% p is final prediction vector
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');
h4 = sigmoid([ones(m, 1) h3] * Theta4');
[dummy, p] = max(h4, [], 2);

% =========================================================================


end
