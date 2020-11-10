function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
C_best = C_values(1);
Sigma_best = Sigma_values(1);
Error = 1000;
for i=1:length(C_values)
    C_current = C_values(i);
for j=1:length(Sigma_values)
        Sigma_current= Sigma_values(j);
        model = svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, Sigma_current));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
if err < Error
            C_best = C_current;
            Sigma_best = Sigma_current;
            Error = err;
end;
end;
end;
C = C_best;
sigma = Sigma_best;
disp(C)
disp(sigma)
disp(Error)

% =========================================================================

end
