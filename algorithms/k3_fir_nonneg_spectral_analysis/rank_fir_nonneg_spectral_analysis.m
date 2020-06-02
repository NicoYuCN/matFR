function [rankx]=rank_fir_nonneg_spectral_analysis(X,maxIter,alpha,beta,gamma)
%	X: Rows of vectors of data points  %d*n
%	L: The laplacian matrix. n*n
%   F: the cluster result  %n*c
%   W: the feature selection matrix d*c
% Please refer to 
%   Li, Z., Yang, Y., Liu, J., Zhou, X. and Lu, H., 2012, July. 
%       Unsupervised feature selection using nonnegative spectral analysis. 
%           In Twenty-Sixth AAAI Conference on Artificial Intelligence.
% ---------------------------------------------------------------
if nargin < 2
    maxIter = 100;
end

if nargin < 3
    alpha = 0.1;
end

if nargin < 4
    beta = 1;
end

if nargin < 5
    gamma = 1000;
end

if nargin == 0
    return; 
end

% ------------------------------------
X = X'; % dim*num
c = 4;
num = size(X,2);
k = 5;
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);
A = zeros(num);
for i = 1:num
    di = distX1(i,2:k+2);
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

A0 = (A+A')/2;

[Label,L1,~] = sc(A0,c);
L = full(L1);

% -----------------------------------------
tLabel = zeros(num,c);
for i = 1:num
    tLabel(i,Label(i)) = 1;
end
F = tLabel;

W = zeros(size(X,1),c);
% -----------------------------------------
[~,nSamp] = size(X);

if size(L,1) ~= nSamp
    error('L is error');
end
XX=X*X';

Wi = sqrt(sum(W.*W,2)+eps);
d = 0.5./Wi;
D = diag(d);

% G=inv(XX+beta*D);
% W=G*X*F;
% Wi = sqrt(sum(W.*W,2)+eps);
% d = 0.5./Wi;
% D = diag(d);
% clear Wi
% M=L+alpha*(eye(nSamp)-X'*G*X);
% clear G
% M=(M+M')/2;
% F = F.*(gamma*F + eps)./(M*F + gamma*F*F'*F + eps);
% F = F*diag(sqrt(1./(diag(F'*F)+eps)));

iter=1;
while iter<=maxIter %|| (iter>2&& obj(end-1)-obj(end)>10^(-3)*obj(end))
    
    G=inv(XX+beta*D);
    W=G*X*F;
    Wi = sqrt(sum(W.*W,2)+eps);
    d = 0.5./Wi;
    D = diag(d);
    clear Wi
    M=L+alpha*(eye(nSamp)-X'*G*X);
    clear G
    M=(M+M')/2;

    F = F.*(gamma*F + eps)./(M*F + gamma*F*F'*F + eps);
    F = F*diag(sqrt(1./(diag(F'*F)+eps)));
    clear Wnew   
    
    obj(iter)=trace(F'*M*F)+gamma/4*norm(F'*F-eye(size(F,2)),'fro')^2;
%     fprintf('Iter %d\tobj=%f\n',iter,obj(end));
    iter=iter+1;
    
end

[~, ranking] = sort(sum(W.*W,2),'descend');
rankx = ranking';