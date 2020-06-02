function rankx = rank_mat_relieff( X,Y, numNeighbor )
%  ------------------------------------------------------------------------
% Please refer to
%   Kononenko, I., ?imec, E. and Robnik-?ikonja, M., 1997. Overcoming the 
%       myopia of inductive learning algorithms with RELIEFF. Applied 
%           Intelligence, 7(1), pp.39-55.
%  ------------------------------------------------------------------------
if nargin < 3
    numNeighbor = 8; % numNeighbor nearest neighbors
end
rankx = relieff( X, Y, numNeighbor);

