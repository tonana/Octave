%% 
%  Assignment 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the linear
%  assignment. You will need to complete the following functions in
%  this assignment:
%
%     warmUpExercise.m
%     plotData.m
%     computeCost.m
%     gradientDescent.m
%     featureNormalize.m
%     computeCostMulti.m
%     gradientDescentMulti.m
%     normalEqn.m
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%d %d], y = %d \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 bedrooms house.
%
% Hint: By using the 'hold on' command, you can plot multiple graphs
%       on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Try different alpha values
% e.g. 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 1.1, 1.3, 2
alpha = 0.1;         % Original 0.01 ---- choose 0.1

% Try different number of iterations
% e.g. 400, 100, 50, 20, 10
num_iters = 50;   % Original 400  ---- choose 50

% Init theta and run gradient descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
% [theta, J1] = gradientDescentMulti(X, y, theta, 0.01, num_iters);
% [theta, J2] = gradientDescentMulti(X, y, theta, 0.03, num_iters);
% [theta, J3] = gradientDescentMulti(X, y, theta, 0.1 , num_iters);

% Plot the convergence graph
figure;
% numel - returns the number of elements of J_history
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% More plots
% plot(1:50, J1(1:50), 'b');
% hold on;
% plot(1:50, J2(1:50), 'r');
% plot(1:50, J3(1:50), 'k');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %14.6f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 bedrooms house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.


% You should change this
% price = [1 (1650 - mu(1)) / sigma(1) (3 - mu(2)) / sigma(2)] * theta;
% Will do the same as the line above
price = [1, ([1650, 3] - mu) ./ sigma] * theta;


% ============================================================

fprintf(['Predicted price: house 1650 ft2, 3 rooms ' ...
         '(using gradient descent):\n $%.2f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 bedrooms house.
%

%% Load data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %14.6f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 bedrooms house
% ====================== YOUR CODE HERE ======================

price = [1 1650 3] * theta; % You should change this


% ============================================================

fprintf(['Predicted price: house 1650 ft2, 3 rooms ' ...
         '(using normal equations):\n $%.2f\n'], price);
