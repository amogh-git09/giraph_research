clear ; close all; clc

input_layer_size  = 2;
hidden_layer_size = 2;
num_labels = 1;
learning_rate = 0.05;

load('researchdata1.mat');
m = size(X, 1);

load('researchweights.mat');

% Unroll parameters
nn_params = [MyTheta1(:) ; MyTheta2(:)];

fprintf('\nFeedforward Using Neural Network ...\n')

lambda = 0;

fprintf(['Cost at parameters (loaded from ex4weights): %f\n'], J);

max_iter = 50

for i = 1:150
  % fprintf('\nBeginning new iteration\n--------------\n');
  [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

  cost = J
  nn_params = nn_params - learning_rate*grad;

  MyTheta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
  MyTheta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
endfor
