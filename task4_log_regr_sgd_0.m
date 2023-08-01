clear all;
close all;
clc;


% Note: Remember to Change class labels {1,2} to values for y appropriate for log-regr => y={0,1}

%rng(12345) % Random number generator seed.

rng_seeds = [2465 114951384 123059 97806 1234 754 48 4844 345709 12345];
costs=[];
times_taken=[];
batch_sizes = [1,10,20,50,100,500];
batch_times = [];

for batch_size = 20
    
    for seed = rng_seeds
    
    rng(seed)
    % Create training/val data
    n_samples = 1000;
    [X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
    y_train_val = (class_labels_train_val==1)*(0) + (class_labels_train_val==2)*(1); % If class 1, y=0. If class 2, y=1.
    regression_targets_train_val = NaN;  % Wont be used for classification.
    % concat 1 for bias
    X_train_val = cat(2, ones(n_samples,1), X_train_val);

    % Divide the data as before. Same function as Task 2.
    [X_train,y_train,X_val,y_val]=divide_data(X_train_val,y_train_val);


    % Create testing data
    n_samples_test = 20000;
    [X_test, regression_targets_test, class_labels_test] = create_data(n_samples_test);
    y_test = (class_labels_test==1)*(0) + (class_labels_test==2)*(1); % If class 1, y=0. If class 2, y=1.
    regression_targets_test = NaN;  % Wont be used for classification.
    % concat 1 for bias
    X_test = cat(2, ones(n_samples_test,1), X_test);
    tic

    % Optimize - Logistic Regression - Stochastic Gradient Descent

    % Set hyper-parameters of learning process here
    learning_rate = 1;
    n_iters = 5000;
    %batch_size = 100;

    
    [cost_vector,theta_opt] = logistic_regression_sgd(X_train, y_train, batch_size, learning_rate, n_iters);
    costs(:,end+1)=cost_vector;
    
    mean_cost = mean(costs,2);
    plot(mean_cost)
    xlabel('No. of iterations completed')
    ylabel('Mean log-loss achieved by model')
    
    % Print out the classification error percentage
    e = classif_error(y_val,log_regr(X_val,theta_opt));
    times_taken(end+1)=toc;
    
    end

    batch_times(end+1) = mean(times_taken);
end




% Using code from Task 5 to plot the data and boundary
[alpha,beta] = boundary(theta_opt,X_train)
hold on
plot_points(X_train_val)


legend('boundary','Class 1', 'Class 2')
ylabel('x_2')
xlabel('x_1')


%Function definitions below

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

function [cost_vector,theta_opt] = logistic_regression_sgd(X_train, y_train, batch_size, learning_rate, iters_total)
    
    % Implements logistic regression using SGD
    
    n = length(X_train);
    
    % Initialize theta
    theta_curr = zeros(3,1);  % Current theta
    
    % Initialise cost array
    cost_vector = zeros(iters_total,1);
   
    for i = 1:iters_total
        
        % Will go sample by sample
        
        gradient = 0;               % Reset gradient each time as we sum it over each sample
        cost_per_sample = zeros(batch_size,1);
        batch = randsample(n,batch_size);


        % Take random batch of data for gradient and loss computation
        X_batch = X_train(batch,:);
        y_batch = y_train(batch);
        
        for idx = 1:batch_size
            
            % Extract the sample
            x = X_batch(idx,:);
            y = y_batch(idx);
            
            % Compute gradients
            y_pred = log_regr(x,theta_curr);
            gradient=gradient+(y_pred-y)*x;

            % Compute cost
            cost_per_sample(idx) = mean_logloss(x,y,theta_curr);
        end
        % Update theta
        gradient = gradient/batch_size;
        theta_curr = theta_curr - learning_rate*gradient';
        
        % Compute cost
        cost_vector(i) = mean(cost_per_sample);
    end
    theta_opt = theta_curr;
%     plot(cost_vector)
%     xlabel('No. of iterations completed')
%     ylabel('Mean log-loss achieved by model')
end

function mean_logloss = mean_logloss(x, y_real, theta)
    % Finds log-loss for a sample
    y_pred = log_regr(x,theta);
    mean_logloss = -y_real*log(y_pred)-(1-y_real)*log(1-y_pred);
end

function y_pred = log_regr(X, theta)
    % Implements the sigmoid function
    
    y_pred = 1./(1+exp(-X*theta));
end

function err_perc = classif_error(y_real, y_pred)
    % Gives error percentage of model
    
    wrong = 0;
    
    for idx = 1:length(y_real)
        if y_pred(idx)>0.5
            y_pred(idx) = 1;
        else
            y_pred(idx)=0;
        end
            
       if y_pred(idx)~=y_real(idx)
           wrong = wrong + 1;
       end
    end
    
    err_perc = wrong*100/length(y_real);
    
end

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

