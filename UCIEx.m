clear; clc;

% Dataset from UCI Machine Learning Repository (http://archive.ics.uci.edu/ml)
n = 1e3;
A1 = readmatrix('covtype.csv'); % Covertype dataset
A2 = readmatrix('Frogs_MFCCs.csv'); % Anuran Calls (MFCC) dataset

X1 = A1(randperm(size(A1,1),n),1:end-1);
X2 = A2(randperm(size(A2,1),n),1:end-4);

% normalize X1 to mean 0 and var 1
d1 = size(X1,2);
means1 = zeros(1,d1);
means1(1:10) = mean(X1(:,1:10));
X1 = X1 - repmat(means1,n,1);
cnorms1 = 1/sqrt(n)*ones(1,d1);
cnorms1(1:10) = 1./sqrt(sum(X1(:,1:10).^2,1));
cnorms1(find(cnorms1 == Inf)) = 0;
X1 = X1*diag(cnorms1);

% normalize X2 to mean 0 and var 1
d2 = size(X2,2);
means2 = mean(X2);
X2 = X2 - repmat(means2,n,1);
cnorms2 = 1./sqrt(sum(X2.^2,1));
cnorms2(find(cnorms2 == Inf)) = 0;
X2 = X2*diag(cnorms2);

% Kernel matrices - Multiquadric, Sigmoid, Thin Plate
K1 = Multiquadric(X1);
K2 = Sigmoid(X2);

% simulation
r = (50:50:500)'; rx = size(r,1);

% singular values of K1 and K2
S1 = svd(K1);
S2 = svd(K2);
NysErr1 = zeros(rx,3); % Error in nuclear norm
SVDErr1 = zeros(rx,1);
NysErr2 = zeros(rx,3); % Error in nuclear norm
SVDErr2 = zeros(rx,1);

% simulation
for i = 1:rx
    SVDErr1(i) = sum(abs(S1(r(i)+1:end))); % best nuclear norm error
    
    [C,W] = Nystrom(K1,r(i),1); % SRFT - no oversampling
    NysErr1(i,1) = sum(svd(K1-C*W*C')); 
    
    [C,W] = IndefNys(K1,r(i)*1.2,r(i),1); % SRFT - oversampling by factor 1.2
    NysErr1(i,2) = sum(svd(K1-C*W*C'));
    
    [C,W] = IndefNys(K1,r(i)*1.2,r(i),2); % uniform col. samp. - oversampling by factor 1.2
    NysErr1(i,3) = sum(svd(K1-C*W*C'));
    
    SVDErr2(i) = sum(abs(S2(r(i)+1:end))); % best nuclear norm error
    
    [C,W] = Nystrom(K2,r(i),1); % SRFT - no oversampling
    NysErr2(i,1) = sum(svd(K2-C*W*C')); 
    
    [C,W] = IndefNys(K2,r(i)*1.2,r(i),1); % SRFT - oversampling by factor 1.2
    NysErr2(i,2) = sum(svd(K2-C*W*C'));
    
    [C,W] = IndefNys(K2,r(i)*1.2,r(i),2); % uniform col. samp. - oversampling by factor 1.2
    NysErr2(i,3) = sum(svd(K2-C*W*C'));
end

% plotting
figure
semilogy(r,SVDErr1), hold on, grid on
semilogy(r,NysErr1)
legend('SVD','SRFT - no oversampling','SRFT - oversampling by factor 1.2','uniform col. samp. - oversampling by factor 1.2')

figure
semilogy(r,SVDErr2), hold on, grid on
semilogy(r,NysErr2)
legend('SVD','SRFT - no oversampling','SRFT - oversampling by factor 1.2','uniform col. samp. - oversampling by factor 1.2')