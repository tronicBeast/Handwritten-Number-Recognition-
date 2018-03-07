function [all_theta] = find_theta(X, y, num_labels, lambda)
%find_theta trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);


options = optimset('GradObj', 'on', 'MaxIter', 50);

for i = 1:num_labels
  all_theta(i,:) = fminunc(@(t)(costfunction(t,X,(y==i),lambda)),initial_theta,options);
end

end
