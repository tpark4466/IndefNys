function K = Multiquadric(X)
% Dataset X, Multiquadric kernel
% Kernel matrix K
    rnorm = sum(X.^2,2);
    K = rnorm+rnorm'-2*(X*X');
    K = sqrt(1+K);
end