function [C,W] = IndefNys(A,s,r,option)

% Input:
% * A - nxn matrix
% * s - sketch size
% * r - target rank
% * option: 0 - Gaussian sketch, 1 - SRTT (DCT) sketch, 2 - uniform col. sampling


% Output
% * [V0,D0] = eig(X^TC); V = V0(:,1:r),D = D0(1:r,1:r), 
% * C = A*X*V;
% * W = inv(D);
% A ~ CWC^T

n = size(A,2);

if nargin < 4
    option = 0;
end

if option == 0
    X = randn(n,s);
    C = A*X;
    [V,D] = eig(X'*C);
    [~,IA] = maxk(abs(diag(D)),r);
    C = C*V(:,IA);
    W = pinv(D(IA,IA));
elseif option == 1
    Dsig = sign(randn(n,1));
    IX = randsample(n,s);
    C = dct(Dsig.*A); C(1,:) = C(1,:)/sqrt(2); 
    C = C(IX,:)*sqrt(n/(s)); C = C';
    W = dct(Dsig.*C); W(1,:) = W(1,:)/sqrt(2);
    W = W(IX,:)*sqrt(n/(s));
    [V,D] = eig(W);
    [~,IA] = maxk(abs(diag(D)),r);
    C = C*V(:,IA);
    W = pinv(D(IA,IA));
else
    II = randsample(n,s);
    C = A(:,II);
    [V,D] = eig(C(II,:));
    [~,IA] = maxk(abs(diag(D)),r);
    C = C*V(:,IA);
    W = pinv(D(IA,IA));
end
end

