function rankx = rank_mat_entropy( X,Y )
%  ------------------------------------------------------------------------
% Please refer to
%   Kullback, S. and Leibler, R.A., 1951. On information and sufficiency. 
%       The annals of mathematical statistics, 22(1), pp.79-86.
%  ------------------------------------------------------------------------

rankx = rankfeatures( X', Y', 'Criterion', 'entropy' );
rankx = rankx';

