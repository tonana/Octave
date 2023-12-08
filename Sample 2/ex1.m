%% 
%  Assignment 1: Linear Regression with one variable
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
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
%  clear     - delete all variables
%  close all - close all figures/plots
%  clc       - clear the terminal
clear; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m 
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot data
% Note: You have to complete the code in plotData.m
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Gradient Descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), X]; % Add a column of ones (Bias Terms) to X
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% Compute and display initial cost
computeCost(X, y, theta)

% Run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% Print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:, 2), X * theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000.
% Remember, the values on the examples are all divided by 10,000.
predict1 = [1, 3.5] * theta;
predict2 = [1, 7.0] * theta;

fprintf('For population = 35,000, we predict a profit of %8.2f\n',
		    predict1 * 10000);
fprintf('For population = 70,000, we predict a profit of %8.2f\n',
        predict2 * 10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace( -1,  4, 100);

% Initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
  for j = 1:length(theta1_vals)
		t = [theta0_vals(i); theta1_vals(j)];    
		J_vals(i, j) = computeCost(X, y, t);
  end
end

% Because of the way meshgrids work in the surf command, 
% we need to transpose J_vals before calling surf, or 
% else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
colormap ("jet");
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);