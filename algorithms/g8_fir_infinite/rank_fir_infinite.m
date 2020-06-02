function [RANKED, WEIGHT] = rank_fir_infinite( X_train, Y_train, alpha, supervision, verbose )
% [RANKED, WEIGHT ] = infFS( X_train, Y_train, verbose ) computes ranks and weights
% of features for input data matrix X_train and labels Y_train using Inf-FS algorithm.
%
% Version 4.0, August 2016.
%
% INPUT:
%
% X_train is a T by n matrix, where T is the number of samples and n the number
% of features.
% Y_train is column vector with class labels (e.g., 0, 1)
% Verbose, boolean variable [0, 1]
%
% OUTPUT:
%
% RANKED are indices of columns in X_train ordered by attribute importance,
% meaning RANKED(1) is the index of the most important/relevant feature.
% WEIGHT are attribute weights with large positive weights assigned
% to important attributes.
%
%  Note, If you use our code or method, please cite our paper:
%  BibTex
%  ------------------------------------------------------------------------
%If you use our toolbox (or method included in it), please consider to cite:

%[1] Roffo, G., Melzi, S., Castellani, U. and Vinciarelli, A., 2017. Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach. arXiv preprint arXiv:1707.07538.

%[2] Roffo, G., Melzi, S. and Cristani, M., 2015. Infinite feature selection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 4202-4210).

%[3] Roffo, G. and Melzi, S., 2017, July. Ranking to learn: Feature ranking and selection via eigenvector centrality. In New Frontiers in Mining Complex Patterns: 5th International Workshop, NFMCP 2016, Held in Conjunction with ECML-PKDD 2016, Riva del Garda, Italy, September 19, 2016, Revised Selected Papers (Vol. 10312, p. 19). Springer.

%[4] Roffo, G., 2017. Ranking to Learn and Learning to Rank: On the Role of Ranking in Pattern Recognition Applications. arXiv preprint arXiv:1706.05933.
%  ------------------------------------------------------------------------
fprintf('\n+ Feature selection method: inf-FS 2016\n');

if (nargin<5)
    verbose = 0;
end
if (nargin<4)
    supervision = 1; % supervised learning
end
if (nargin<3)
    alpha = 0.5; % default, it should be cross-validated.
end

%% The Inf-FS method

%% 1) Standard Deviation over the samples
if (verbose)
    fprintf('1) Priors/weights estimation \n');
end
if supervision
    s_n = X_train(Y_train==0,:); % Binary 0/1
    s_p = X_train(Y_train==1,:);
    mu_s_n = mean(s_n);
    mu_s_p = mean(s_p);
    priors_corr = ([mu_s_p - mu_s_n].^2);
    st   = std(s_p).^2;
    st   = st+std(s_n).^2;
    st(find(st==0))=10000;  % remove ones where nothing occurs
    corr_ij = priors_corr ./ st;
    corr_ij = [corr_ij'*corr_ij];
    corr_ij = corr_ij - min(min( corr_ij ));
    corr_ij = corr_ij./max(max( corr_ij )); % values in [0,1]
else
    
    [ corr_ij, pval ] = corr( X_train, 'type','Spearman' );
    corr_ij(isnan(corr_ij)) = 0; % remove NaN
    corr_ij(isinf(corr_ij)) = 0; % remove inf
    corr_ij =  1-abs(corr_ij);
end
% Standard Deviation Est.
STD = std(X_train,[],1);
STDMatrix = bsxfun( @max, STD, STD' );
STDMatrix = STDMatrix - min(min( STDMatrix ));
sigma_ij = STDMatrix./max(max( STDMatrix ));
sigma_ij(isnan(sigma_ij)) = 0; % remove NaN
sigma_ij(isinf(sigma_ij)) = 0; % remove inf

%% 2) Building the graph G = <V,E>
if (verbose)
    fprintf('2) Building the graph G = <V,E> \n');
end
A =  ( alpha*corr_ij + (1-alpha)*sigma_ij );

factor = 0.99;

%% 3) Letting paths tend to infinite: Inf-FS Core
if (verbose)
    fprintf('3) Letting paths tend to infinite \n');
end
I = eye( size( A ,1 )); % Identity Matrix

r = (factor/max(eig( A + eps))); % Set a meaningful value for r

y = I - ( r * A );

S = inv( y ) - I; % see Gelfand's formula - convergence of the geometric series of matrices

%% 4) Estimating energy scores
if (verbose)
    fprintf('4) Estimating energy scores \n');
end
WEIGHT = sum( S , 2 ); % energy scores s(i)

%% 5) Ranking features according to s
if (verbose)
    fprintf('5) Features ranking  \n');
end
[~ , RANKED ]= sort( WEIGHT , 'descend' );

RANKED = RANKED';
WEIGHT = WEIGHT';

end

%  =========================================================================
%   More details:
%   Reference   : Infinite Feature Selection
%   Author      : Giorgio Roffo and Simone Melzi and Marco Cristani
%   Link        : http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Roffo_Infinite_Feature_Selection_ICCV_2015_paper.html
%   ResearchGate: https://www.researchgate.net/publication/282576688_Infinite_Feature_Selection
%  =========================================================================

