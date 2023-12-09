%% 
%  Assignment 2: Logistic Regression with Regularization
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the second part
%  of the assignment which covers regularization with logistic regression.
%
%  You will need to complete the following function in this assignment:
%
%     costFunctionReg.m
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features
fprintf('Number of Features, including the Intercept Term\n');
% "+1" to include X0 in the counting
fprintf('  Before Polynomial Expansion : %2d\n', size(X, 2) + 1);
% mapFeature will add the intercept term for you
X = mapFeature(X(:,1), X(:,2));
fprintf('  After  Polynomial Expansion : %2d\n', size(X, 2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized
% logistic regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision boundary.
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda?
%  How does the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;  % Try 0, 1, 10, 100

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t) costFunctionReg(t, X, y, lambda), initial_theta, options);
	
% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n\n', mean(p == y) * 100);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           %
% Extra Exercise            %
%                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Do a prediction for the Example:
%       X1        X2  y
% -0.13882, -0.27266, 1
fprintf('\n\nPredicting one value\n');
% Feature    X1 = ?
% Feature    X2 = ?
% Prediction p  = ?

% Do a prediction for the Examples:
%       X1        X2  y
% -0.21947  -0.01681  1
% -0.13882  -0.27266  1
%  0.18376   0.93348  0
fprintf('\n\nPredicting three values\n');
% Feature    X1 = ?
% Feature    X2 = ?
% Prediction p  = ?
