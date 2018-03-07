function p = predictall(all_theta, X)
%PREDICTALL Predict the label for a all images in a dataset.

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

for j=1:m
	p(j,:) = max(X(j ,:) * (all_theta'));
  for i=1:size((X(j,:)*(all_theta')),2)
    if(max(X(j ,:) * (all_theta'))==(X(j,:)*(all_theta'))(1,i))
      p(j,:)=i;
    end
  end
end

end