function [ranking] = rank_fir_KruskalWallis(X, Y)
%  ------------------------------------------------------------------------
% Please refer to
%   Hollander, M., Wolfe, D.A. and Chicken, E., 2013. Nonparametric 
%       statistical methods (Vol. 751). John Wiley & Sons.
%  ------------------------------------------------------------------------

[~, n] = size(X);
out.W = zeros(n,1);

for i=1:n
    out.W(i) = -kruskalwallis(vertcat(X(:,i)', Y'),{},'off');
end

[~, ranking] = sort(out.W, 'descend');
ranking = ranking';
end