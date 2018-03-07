function [J, grad] = costfunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost and gradient for logistic regression 

m = length(y);
 
J = 0;
grad = zeros(size(theta));


h=sigmoid(X*theta);
J=(1/m)*sum(((-1)*y).*log(h)-(((-1)*y+1).*log((-1)*h+1)));
J=J+lambda*(1/(2*m))*(sum(theta.*theta)-theta(1,1)*theta(1,1));
g=(1/m)*(X')*(h-y);
grad = g+(lambda*(1/m).*theta);
grad(1,:) = grad(1,:) - (lambda*(1/m).*theta(1,:));

grad = grad(:);

end
