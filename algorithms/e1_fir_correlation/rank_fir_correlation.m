function rankx = rank_fir_correlation(X)
%  ------------------------------------------------------------------------
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
% Please refer to
%   Roffo, G., 2016. Feature selection library (MATLAB toolbox). 
%       arXiv preprint arXiv:1607.01327.
%  ------------------------------------------------------------------------
corrMatrix = abs( corr(X) );

% Ranking according to minimum correlations
scores = min(corrMatrix,[],2);


[~,rankx] = sort(scores,'ascend');
rankx = rankx';
end