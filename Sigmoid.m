function K = Sigmoid(X)
% Dataset X, Sigmoid kernel
% Kernel matrix K
    K = X*X';
    K = tanh(1+K);
end