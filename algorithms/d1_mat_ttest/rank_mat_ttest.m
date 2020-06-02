function rankx = rank_mat_ttest( X,Y )
%  ------------------------------------------------------------------------
% Please refer to
%   Box, J.F., 1987. Guinness, Gosset, Fisher, and small samples. 
%       Statistical science, 2(1), pp.45-52.
%  ------------------------------------------------------------------------

rankx = rankfeatures( X', Y', 'Criterion', 'ttest' );
rankx = rankx';

