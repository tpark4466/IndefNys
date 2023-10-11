function [C,W] = Nystrom(A,s,option)

% Input:
% * A - nxn matrix
% * s - sketch size
% * option: 0 - Gaussian sketch, 1 - SRTT (DCT) sketch, 2 - uniform col. sampling

% Output
% * [V,D] = eig(X^TC); 
% * C = A*X*V;
% * W = inv(D);
% A ~ CWC^T

n = size(A,2);

if nargin < 3
    option = 0;
end

if option == 0
    X = randn(n,s);
    C = A*X;
    W = X'*C;
    [V,D] = eig(W);
    C = C*V;
    W = pinv(D);
elseif option == 1
    Dsig = sign(randn(n,1));
    IX = randsample(n,s);
    C = dct(Dsig.*A); C(1,:) = C(1,:)/sqrt(2); 
    C = C(IX,:)*sqrt(n/(s)); C = C';
    W = dct(Dsig.*C); W(1,:) = W(1,:)/sqrt(2);
    W = W(IX,:)*sqrt(n/(s));
    [V,D] = eig(W);
    C = C*V;
    W = pinv(D);
else
    II = randsample(n,s);
    C = A(:,II);
    [V,D] = eig(C(II,:));
    C = C*V;
    W = pinv(D);
end
end

