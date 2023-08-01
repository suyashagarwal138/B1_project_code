clear all;
close all;
clc;


% Note: Remember to Change class labels {1,2} to values for y appropriate for SVM => y={-1,+1}

rng(12345) % Random number generator seed.

% Create training data
n_samples = 1000;
[X_train, regression_targets_train, class_labels_train] = create_data(n_samples);
y_train = (class_labels_train==1)*(-1) + (class_labels_train==2)*(1); % If class 1, y=-1. If class 2, y=+1
regression_targets_train = NaN;  % Wont be used for classification.
% concat 1 for bias
X_train = cat(2, ones(n_samples,1), X_train);


% Create testing data
n_samples_test = 1000;
[X_test, regression_targets_test, class_labels_test] = create_data(n_samples_test);
y_test = (class_labels_test==1)*(-1) + (class_labels_test==2)*(1); % If class 1, y=-1. If class 2, y=1.
regression_targets_test = NaN;  % Wont be used for classification.
% concat 1 for bias
X_test = cat(2, ones(n_samples_test,1), X_test);


% Optimize - Support Vector Machine - Linear Programming
theta_opt = train_SVM_linear_progr(X_train, y_train)


e = classif_error(y_test, svm(X_test,theta_opt));

% Plot the decision boundary by calling this function
[alpha,beta] = boundary(theta_opt,X_train)
hold on

% This function plots the input data so we can contextualise the decision
% boundary.
plot_points(X_train)

legend('Linear programming','Logistic regression','Class 1', 'Class 2')
ylabel('x_2')
xlabel('x_1')
% Function definitions below


function theta_opt = train_SVM_linear_progr(X_train, y_train)

    % Finds optimum theta using linear programming
    
    n = length(X_train);                % Set n to no. of samples for convenience
    
    y_matrix = diag(y_train);           % So we can easily multiply with X_train 
    iden = eye(n);
    
    A = -[y_matrix*X_train iden];
    b = - ones(n,1);                    % A and b define our constraint
    
    f = [0;0;0;ones(n,1)];              % Objective function sums slack values
    
    Aeq = [];
    beq = [];
    ub = [];
    
    lb = [-Inf;-Inf;-Inf;zeros(n,1)];   % No lower bounds on theta, 0 for slack
    
    psi_opt = linprog(f,A,b,Aeq,beq,lb,ub);
    
    theta_opt = psi_opt(1:3);
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
    
    y_linprog = alpha*X(:,2)+beta;
    
    plot(X(:,2),y_linprog,'g')
    
    % For the graph in the report, modified this function with code below
    hold on
    alpha2 = -0.7642;       % Values from GD
    beta2=2.6247;
    y_GD = alpha2*X(:,2)+beta2;
    plot(X(:,2),y_GD,'m')

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

