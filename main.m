clear ; close all; clc

input_layer_size  = 400;
num_labels = 10;
% (we have mapped "0" to label 10)

%%  Loading and Visualizing Data

fprintf('Loading and Visualizing Data ...\n');

load('imgdata.mat'); % training data stored in arrays X, y
m = size(X, 1);

fprintf('Total Number of images in data set: %d\n',m);

fprintf('Total 2000 Images are randomly choosen for training\n');

% Randomly select 100 data points to display
rand_indices = randperm(m);
disp(m);
sel = X(rand_indices(1:100), :);
sel_y = y(rand_indices(1:100), :);

display(sel);

sel = X(rand_indices(1:2000), :);
sel_y = y(rand_indices(1:2000), :);

fprintf('Program paused. Press enter to continue.\n');
pause;



%% One-vs-All Training
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = find_theta(sel, sel_y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Predict for One-Vs-All

pred = predictall(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
random_matrix=randperm(m);
for i = 1:m
  display(X(random_matrix(i),:));
  [output] = predictOne(all_theta,X(random_matrix(i),:));
  fprintf('Actual Answer: ');
  disp(output);
  s = input('Paused - press enter to continue, q to exit:','s');
  if s == 'q'
    break
  end
end

