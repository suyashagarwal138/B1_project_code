clear all;
close all;
clc;


%rng(12345) % Random number generator seed.


%... rest is up to you ...
%...

% List of RNG seeds we will use for averaging over
rng_seeds = [12345 114951384 123059 97806 1234 754 48 4844 345709 02465];
% I changed the numbers in this array from the version of the code which I used to collect
% results in my report. Therefore, this code will not get exactly the same
% numbers but everything else is the same. 

% Sets of training dataset sizes for different experiments
n_samples = [4,10,20,100,1000,10000];       % For testing MSE
n_samples2 = [10,100,1000,10000];           % For testing runtime

% These will collect the relevant summary statistics automatically
mean_mse_vals = [];                         
mse_std_vals = [];
mean_times = [];


for n_samples_train=n_samples2
    
    % Initialise this array for each training dataset size as we are only
    % interested in mean and std values which are collected after each
    % loop.
    mse_values = [];
    
    % Collect runtimes here
    times_taken = [];

    for seed = rng_seeds
        % This for loop is for averaging over 10 different seeds.
        
        rng(seed)

        % Create training data
        [X_train, regression_targets_train, class_labels_train] = create_data(n_samples_train);
        y_train = regression_targets_train;  % Target for regression model - what to predict
        class_labels_train = NaN;  % Wont be used for regression.
        % concat 1 for bias
        X_train = cat(2, ones(n_samples_train,1), X_train);

        % Create testing data
        n_samples_test = 1000;
        [X_test, regression_targets_test, class_labels_test] = create_data(n_samples_test);
        y_test = regression_targets_test;
        class_labels_test = NaN;  % Wont be used for regression.
        % concat 1 for bias
        X_test = cat(2, ones(n_samples_test,1), X_test);


        % Return theta_opt
        tic
        theta_opt = mse_regression_closed_form(X_train, y_train);
        times_taken(end+1) = toc;

        % Find the MSE that optimum theta gives
        mse_values(end+1) = mean_squared_error(X_test, y_test, theta_opt);
    
    end
    
    mean_mse_vals(end+1)=mean(mse_values);
    mse_std_vals(end+1)=std(mse_values);
    mean_times(end+1)=mean(times_taken);
end


% Below is the code from the initial run, before adding for loops,
% all commented out

% 
% % Return theta_opt
% tic
% theta_opt = mse_regression_closed_form(X_train, y_train)
% time_taken = toc;
% 
% % Find the MSE that optimum theta gives
% mse = mean_squared_error(X_train, y_train, theta_opt)




% Function definitions below


function mse = mean_squared_error(X, y, theta)
    
    n = length(X);
    mse = 1/n * (X*theta-y)'*(X*theta-y);
end

function theta_opt = mse_regression_closed_form(X, y)
    % implement
    theta_opt = inv(X'*X)*X'*y;
end




