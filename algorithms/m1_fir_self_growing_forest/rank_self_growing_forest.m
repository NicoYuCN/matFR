function [rankx, forest, featimp] = rank_self_growing_forest(X,Y)
% Jan 17, 2025
%   Feature-Ranking Self-Growing Forest algorithm (FSF) 
%           for classification and for regression
%
% --------------------------------------------------------------------
if nargin < 2
    rankx = [];
    forest = [];
    featimp = [];
    return;
end


[forest,featimp] = FSF(X,Y);

[featimp, rankx ] = sort(featimp, 'descend');

end

