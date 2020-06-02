function rankx = rank_mat_lasso( X,Y, lambda )
%  ------------------------------------------------------------------------
% Please refer to
%   Tibshirani, R., 1996. Regression shrinkage and selection via the lasso.
%       Journal of the Royal Statistical Society: Series B (Methodological), 
%           58(1), pp.267-288.
%  ------------------------------------------------------------------------
if nargin < 3
    lambda = 16;
end

B = lasso( X, Y );
[ ~, rankx ] = sort( B( :, lambda ), 'ascend' );
rankx = rankx';

