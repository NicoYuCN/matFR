function [rankx] = rank_fir_fisher(X,Y)
%  ------------------------------------------------------------------------
% Please refer to
%   Gu, Q., Li, Z. and Han, J., 2012. Generalized fisher score for feature 
%       selection. arXiv preprint arXiv:1202.3725.
%  ------------------------------------------------------------------------
% Fisher Score, use the N var formulation
%   X, the data, each raw is an instance
%   Y, the label in 1 2 3 ... format

numC = max(Y);
[~, numF] = size(X);
out.W = zeros(1,numF);

% statistic for classes
cIDX = cell(numC,1);
n_i = zeros(numC,1);
for j = 1:numC
    cIDX{j} = find(Y(:)==j);
    n_i(j) = length(cIDX{j});
end

% calculate score for each features
for i = 1:numF
    temp1 = 0;
    temp2 = 0;
    f_i = X(:,i);
    u_i = mean(f_i);
    
    for j = 1:numC
        u_cj = mean(f_i(cIDX{j}));
        var_cj = var(f_i(cIDX{j}),1);
        temp1 = temp1 + n_i(j) * (u_cj-u_i)^2;
        temp2 = temp2 + n_i(j) * var_cj;
    end
    
    if temp1 == 0
        out.W(i) = 0;
    else
        if temp2 == 0
            out.W(i) = 100;
        else
            out.W(i) = temp1/temp2;
        end
    end
end

[~, rankx] = sort(out.W, 'descend');
end
