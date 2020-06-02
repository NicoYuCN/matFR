function rankx = rank_mat_wilcoxon( X,Y )
%  ------------------------------------------------------------------------
% Please refer to
%   Wilcoxon, F., 1992. Individual comparisons by ranking methods. 
%       In Breakthroughs in statistics (pp. 196-202). Springer, New York, NY.
%  ------------------------------------------------------------------------
rankx = rankfeatures( X', Y', 'Criterion', 'wilcoxon' );
rankx = rankx';

