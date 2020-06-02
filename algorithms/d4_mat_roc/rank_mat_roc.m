function rankx = rank_mat_roc( X,Y )
%  ------------------------------------------------------------------------
% Please refer to
%   Bradley, A.P., 1997. The use of the area under the ROC curve in the 
%       evaluation of machine learning algorithms. Pattern recognition, 
%           30(7), pp.1145-1159.
%  ------------------------------------------------------------------------

rankx = rankfeatures( X', Y', 'Criterion', 'roc' );
rankx = rankx';

