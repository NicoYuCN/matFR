function [rankx] = rank_joint_embed_learn_sparse_regression(data,ReducedDim, alpha,beta)
% 2011. Feature selection via joint embedding learning and sparse regression
%  Input:
%        data:   nSmp * nFea;
%        W_ori:  the original local similarity matrix
%   ReducedDim:  the dimensionality for low dimensionality embedding $Y$
%        alpha:  an emperical parameter (check the paper)
%         beta:  an emperical parameter (check the paper)
%---------------------------------------------------------------------
if nargin < 2
    ReducedDim = size(data,2);
end

if nargin < 3
    alpha = 2; % 1.5 ~ 2.4
end

if nargin < 4
    beta = 1e-2; % 1e-2 ~ 1e-1
end
%---------------------------------------------------------------------
options.KernelType = 'Gaussian';
options.t = optSigma(data);
W_ori = constructKernel(data,[],options);
%---------------------------------------------------------------------
[nSmp,nFea] = size(data);

% Normalization of W_ori
D_mhalf = full(sum(W_ori,2).^-.5);
W = compute_W(W_ori,data,D_mhalf);

% Eigen_decomposition
Y = compute_Y(data,W, ReducedDim, D_mhalf);
if issparse(data)
    data = [data ones(size(data,1),1)];
    [nSmp,nFea] = size(data);
else
    sampleMean = mean(data);
    data = (data - repmat(sampleMean,nSmp,1));
end

% To minimize squared loss with L21 normalization
% (1) Initialization
AA = data'*data;
Ay = data'*Y;
W_compute = (AA+alpha*eye(nFea))\Ay;
d = sqrt(sum(W_compute.*W_compute,2));

itermax = 20;
obj = zeros(itermax,1);

for iter = 1:itermax
    % fix D to updata W_compute and Y
    D = 2*spdiags(d,0,nFea,nFea);
    % updata Y
    A = (D*data'*data+alpha*eye(nFea));
    Temp  = A\(D*data');
    Temp =  data*Temp;
    Temp = W_ori-beta*eye(nSmp)+beta*Temp;
    
    % normalize
    Temp = compute_W(Temp,data,D_mhalf);
    % eigen_decomposition
    Y = compute_Y(data,Temp, ReducedDim, D_mhalf);
    
    % updata W
    B = D*data'*Y;
    W_compute = A\B;
    
    % fix W and update D
    d = sqrt(sum(W_compute.*W_compute,2));   
end

[ ~, rankx ] = sort( sum( W_compute.*W_compute, 2 ), 'ascend' );
rankx = rankx';
end
