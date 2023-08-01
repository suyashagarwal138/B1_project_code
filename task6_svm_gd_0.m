clear all;
close all;
clc;

% Note: Remember to change class labels {1,2} to values for y appropriate for SVM => y={-1,+1}

rng(12345) % Random number generator seed.

% Create training/val data
n_samples = 1000;
[X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
y_train_val = (class_labels_train_val==1)*(-1) + (class_labels_train_val==2)*(1); % If class 1, y=-1. If class 2, y=+1
regression_targets_train_val = NaN;  % Wont be used for classification.
% concat 1 for bias
X_train_val = cat(2, ones(n_samples,1), X_train_val);

% Divide the data as before. Same function as Task 2.
[X_train,y_train,X_val,y_val]=divide_data(X_train_val,y_train_val);

% Create testing data
n_samples_test = 20000;
[X_test, regression_targets_test, class_labels_test] = create_data(n_samples_test);
y_test = (class_labels_test==1)*(-1) + (class_labels_test==2)*(1); % If class 1, y=-1. If class 2, y=1.
regression_targets_test = NaN;  % Wont be used for classification.
% concat 1 for bias
X_test = cat(2, ones(n_samples_test,1), X_test);

% Optimize - Support Vector Machine - Gradient Descent with Hinge Loss

% Set hyper-parameters of learning process here
learning_rate = 0.1;
n_iters = 10000;


theta_opt = train_SVM_hingeloss_gd(X_train, y_train, learning_rate, n_iters);

e = classif_error(y_val, svm(X_val,theta_opt))

% [alpha,beta] = boundary(theta_opt,X_train)
% hold on
% plot_points(X_train_val)

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

function theta_opt = train_SVM_hingeloss_gd(X_train, y_train, learning_rate, iters_total)
    % Hinge loss gradient descent function to optimise SVM.
        
    n = length(X_train);
    
    % Initialize theta
    theta_curr = zeros(3,1);  % Current theta
    
    % Initialise cost array
    cost_vector = zeros(iters_total,1);

    for i = 1:iters_total
        
        % Will go sample by sample
        
        gradient = 0;               % Reset gradient each time as we sum it over each sample
        cost_per_sample = zeros(n,1);
        
        for idx = 1:n
            
            % Extract the sample
            x = X_train(idx,:);
            y = y_train(idx);
           
            % Compute cost
            cost_per_sample(idx) = hinge_loss_per_sample(x,y,theta_curr);
            
            % Compute gradients
            if cost_per_sample(idx)>0
                 gradient=gradient-y*x;
            end
            
        end
        % Update theta
        gradient = gradient/n;
        theta_curr = theta_curr - learning_rate*gradient';
        
        % Compute cost
        cost_vector(i) = sum(cost_per_sample)/n;
    end
    theta_opt = theta_curr;
    plot(cost_vector)
    xlabel('No. of iterations completed')
    ylabel('Average hinge loss achieved by model')
end

function loss_per_sample = hinge_loss_per_sample(X, y_true, theta)
    loss_per_sample = max(0,1-(y_true*svm(X,theta)));
end

function class_score = svm(X, theta)
    class_score = X*theta;
end

function err_perc = classif_error(y_real, y_pred)
    % Gives error percentage of model
    
    wrong = 0;
    
    for idx = 1:length(y_real)
        if y_pred(idx)>0
            y_pred(idx) = 1;
        else
            y_pred(idx)=-1;
        end
            
       if y_pred(idx)~=y_real(idx)
           wrong = wrong + 1;
       end
    end
    
    err_perc = wrong*100/length(y_real);
    
end

% Functions to define the boundary and plot it against the data
function [alpha,beta] = boundary(theta,X)
    % Plots the decision boundary
    alpha = -theta(2)/theta(3);
    beta = -theta(1)/theta(3);
    
    y = alpha*X(:,2)+beta;
    
    plot(X(:,2),y,'g')

end

function [] = plot_points(X)
    % Takes input data and plots the two classes of points in different
    % colours
    
    n = length(X);
    
    c1 = X(1:n/2,3);
    c2 = X(n/2+1:end,3);
    
    plot(X(1:n/2,2),c1,'.r')
    hold on
    plot(X(n/2+1:end,2),c2,'.b')

end







