clear; clc;

n = 1000;

U = orth(randn(n,n));
S = logspace(0,-10,n).*sign(randn(1,n)); % geometrically decaying sing. val.
%S = [ones(1,100),1e-10*ones(1,n-100)].*sign(randn(1,n)); % gap in sing. val.

A = U*diag(S)*U';

% simulation
r = 40:40:400; rx = size(r,2);
NysErr = zeros(rx,3); % Error in nuclear norm
SVDErr = zeros(rx,1);

for i = 1:rx
    SVDErr(i) = sum(abs(S(r(i)+1:end))); % best nuclear norm error
    
    [C,W] = Nystrom(A,r(i)); % no oversampling
    NysErr(i,1) = sum(svd(A-C*W*C')); 
    
    [C,W] = IndefNys(A,r(i)+5,r(i)); % oversampling by 5
    NysErr(i,2) = sum(svd(A-C*W*C'));
    
    [C,W] = IndefNys(A,r(i)*1.5,r(i)); % oversampling by factor 1.5
    NysErr(i,3) = sum(svd(A-C*W*C'));

end

% plotting
figure
semilogy(r,SVDErr), hold on, grid on
semilogy(r,NysErr)
legend('SVD','no oversampling','oversampling by 5','oversampling by factor 1.5')