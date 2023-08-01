clear all;
close all;
clc;


rng(12345) % Random number generator seed.

% Create training/val data
n_samples = 1000;
[X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
y_train_val = regression_targets_train_val;  % Target for regression model - what to predict
class_labels_train_val = NaN;  % Wont be used for regression.
% concat 1 for bias
X_train_val = cat(2, ones(n_samples,1), X_train_val);


% Divide the data into training and validation sets.
[X_train,y_train,X_val,y_val]=divide_data(X_train_val,y_train_val);


% Create testing data
n_samples_test = 20000;
[X_test, regression_targets_test, class_labels_test] = create_data(n_samples_test);
y_test = regression_targets_test;
class_labels_test = NaN;  % Wont be used for regression.
% concat 1 for bias
X_test = cat(2, ones(n_samples_test,1), X_test);


%... rest is for you to implement ...
%...


% Optimize - Linear Regression - Gradient Descent

% Set hyper-parameters of learning process here
learning_rate = 0.1;
n_iters = 1000;

theta_opt = linear_regression_gd(X_train, y_train, learning_rate, n_iters)

mse = mean_squared_error(X_val, y_val, theta_opt)

% Function definitions below

function [X_train,y_train,X_val,y_val]=divide_data(X_train_val,y_train_val)

    % Takes input data and randomly splits it 80/20 into training and
    % validation data sets.

    n = length(X_train_val);
    idx = randsample(n,n);                      % Randomises the split
    cut_point = 0.8*n;                          % For an 80/20 split
    
    idx_train = idx(1:cut_point);               % Indices for training data
    idx_val = idx(cut_point+1:end);             % Indices for val data
    
    
    X_train = X_train_val(idx_train,:);
    X_val = X_train_val(idx_val,:);
    
    y_train = y_train_val(idx_train,:);
    y_val = y_train_val(idx_val,:);

end



function theta_opt = linear_regression_gd(X_train, y_train, learning_rate, iters_total)
    %....
    
    
    n = length(X_train);
    
    % Initialize theta
    theta_curr = zeros(3,1);  % Current theta

    for i = 1:iters_total
        % Compute gradients
        gradient = (2/n) * (X_train'*X_train*theta_curr-X_train'*y_train);

        % Update theta
        theta_curr = theta_curr - learning_rate*gradient;

        % Compute cost
        cost = mean_squared_error(X_train,y_train,theta_curr);

    end
    theta_opt = theta_curr;
end


function mse = mean_squared_error(X, y, theta)
    
    n = length(X);
    mse = 1/n * (X*theta-y)'*(X*theta-y);
end

