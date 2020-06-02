function [rankx] = rank_fir_laplacian_score(X)
%	Usage:
%	[Y] = LaplacianScore(X)
%
%	X: Rows of vectors of data points
%	Y: Vector of (1-LaplacianScore) for each feature.
%      The features with larger y are more important.
%   rankx: the ranking index
%
%    Examples:
%
%       fea = rand(50,70);
%       options = [];
%       options.Metric = 'Cosine';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'Cosine';
%       W = constructW(fea,options);
% 
%       LaplacianScore = LaplacianScore(fea,W);
%       [junk, index] = sort(-LaplacianScore);
%       
%       newfea = fea(:,index);
%       %the features in newfea will be sorted based on their importance.
%
%	Type "LaplacianScore" for a self-demo.
%
% See also constructW
%
%Reference:
%
%   Xiaofei He, Deng Cai and Partha Niyogi, "Laplacian Score for Feature Selection".
%   Advances in Neural Information Processing Systems 18 (NIPS 2005),
%   Vancouver, Canada, 2005.   
%
% %   Deng Cai, 2004/08
%
%  Version 5.0 August 2017
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
%  ------------------------------------------------------------------------

W = dist( X' );
W = -W./max( W(:) ); % it's a similarity

% -------------------------------
[nSmp,~] = size(X);

if size(W,1) ~= nSmp
    error('W is error');
end

D = full(sum(W,2));
L = W;

% allone = ones(nSmp,1);


tmp1 = D'*X;

D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);

DPrime = sum((X'*D)'.*X)-tmp1.*tmp1/sum(diag(D));
LPrime = sum((X'*L)'.*X)-tmp1.*tmp1/sum(diag(D));

DPrime(find(DPrime < 1e-12)) = 10000;

Y = LPrime./DPrime;
Y = Y';
Y = full(Y);
[~,rankx] = sort(-Y);
rankx = rankx';
