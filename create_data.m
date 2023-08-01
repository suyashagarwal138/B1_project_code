
function [X, regression_targets, class_labels] = create_data(n_samples)
    % Function returns:
    % X: Features of each sample. Matrix of dims: n_samples x features
    % regression_targets: Regression target for each sample. Vector of dims: n_samples x 1
    % class_labels: Class of each sample. Vector of dims: n_samples x 1
    % Row i in each 3 matrix corresponds to the same sample i

    r_noise_var = 0.04; % Variance for regression target
    n_classes = 2;
    n_features = 2;

    % One row per n_classes. One column per n_features
    % 1st row: x1,x2 of the mean of a Gaussian for class 1.
    % 2nd row: x1,x2 of the mean of a Gaussian for class 2.
    x_mu = [1,1;
            2,2];
    % One Gaussian per class.
    % Diagonal covariance matrix for features of samples from class 1
    x_var(:,:,1) = [0.15, 0;
                    0, 0.15];
    % Diagonal covariance matrix for features of samples from class 2
    x_var(:,:,2) = [0.15, 0;
                    0, 0.15];

    n_samples_per_class = floor(n_samples/n_classes);

    for c = 1:n_classes

        if c == n_classes  % This is in case n_classes did not perfectly divide n_samples
            n_samples_per_class = n_samples - n_samples_per_class * (n_classes-1);
        end

        % ---- Sample X ----
        % Multivariate Normal (MVN):
        % mvnpdf(X,mu,Sigma) => https://uk.mathworks.com/help/stats/multivariate-normal-distribution.html
        % Sample from MVN:
        % mvnrnd(mu,Sigma,n_samples) => https://uk.mathworks.com/help/stats/mvnrnd.html

        X_per_c(:,:,c) = mvnrnd(x_mu(c,:), x_var(:,:,c), n_samples_per_class);
        
        % --- Generate Regression target r ----
        regression_targets_per_c(:,c) = 0.3 * X_per_c(:,1,c) + 0.7 * X_per_c(:,2,c) + 2;
        % Add noise
        r_noise = mvnrnd(0, r_noise_var, n_samples_per_class);
        regression_targets_per_c(:,c) = regression_targets_per_c(:,c) + r_noise;
        
        % --- Class of each sample (1,2, ....)
        class_labels_per_c(:,c) = ones(n_samples_per_class,1) * c ;
    end
    
    % Concat samples of all classes along 1st dimension (rows)
    X = X_per_c(:,:,1);
    regression_targets = regression_targets_per_c(:,1);
    class_labels = class_labels_per_c(:,1);
    for c = 2:n_classes
        X = cat(1, X(:,:), X_per_c(:,:,c));
        regression_targets = cat(1, regression_targets(:), regression_targets_per_c(:,c));
        class_labels = cat(1, class_labels(:), class_labels_per_c(:,c));
    end
end

