function K = ThinPlate(X)
% Dataset X, Thin Plate kernel
% Kernel matrix K
    rnorm = sum(X.^2,2);
    K = rnorm+rnorm'-2*(X*X');
    II = logical(abs(K)< 1e-17);
    K(~II) = K(~II).*log(K(~II));
    K(II) = 0;
    K = real(K);
end