function rankx = rank_mat_bhattacharyya( X,Y )
%  ------------------------------------------------------------------------
% Please refer to
%   Chernoff, H., 1952. A measure of asymptotic efficiency for tests of 
%       a hypothesis based on the sum of observations. The Annals of 
%           Mathematical Statistics, 23(4), pp.493-507.
%  ------------------------------------------------------------------------

rankx = rankfeatures( X', Y', 'Criterion', 'bhattacharyya' );
rankx = rankx';

