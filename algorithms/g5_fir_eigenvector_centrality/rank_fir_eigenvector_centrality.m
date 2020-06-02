function [ rankx ] = rank_fir_eigenvector_centrality( X_train, Y_train, alpha )
% [RANKED, WEIGHT ] = ECFS( X_train, Y_train, alpha ) computes ranks and weights
% of features for input data matrix X_train and labels Y_train using EC algorithm.
%
% Version 3.0, August 2016.
% 
% INPUT:
%
% X_train is a T by n matrix, where T is the number of samples and n the number
% of features.
% Y_train is column vector with class labels (e.g., 0, 1)
% Alpha values in [0, 1]
%
% OUTPUT:
%
% RANKED are indices of columns in X_train ordered by attribute importance,
% meaning RANKED(1) is the index of the most important/relevant feature.
% WEIGHT are attribute weights with large positive weights assigned
% to important attributes. 
%
%  Note, If you use our code or method, please cite our paper:
%  BibTex
%  ------------------------------------------------------------------------
% @InProceedings{RoffoECML16, 
% author={G. Roffo and S. Melzi}, 
% booktitle={Proceedings of New Frontiers in Mining Complex Patterns (NFMCP 2016)}, 
% title={Features Selection via Eigenvector Centrality}, 
% year={2016}, 
% keywords={Feature selection;ranking;high dimensionality;data mining}, 
% month={Oct}}
%  ------------------------------------------------------------------------
%  Version 5.0 August 2017
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk

%  If you use our toolbox please cite our supporting papers:
% 
%  BibTex
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV17, 
% author={Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli}, 
% booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach}, 
% year={2017}, 
% month={Oct}}
%  ------------------------------------------------------------------------
% @Inbook{Roffo2017,
% author="Roffo, Giorgio
% and Melzi, Simone",
% editor="Appice, Annalisa
% and Ceci, Michelangelo
% and Loglisci, Corrado
% and Masciari, Elio
% and Ra{\'{s}}, Zbigniew W.",
% title="Ranking to Learn:",
% bookTitle="New Frontiers in Mining Complex Patterns: 5th International Workshop, NFMCP 2016, Held in Conjunction with ECML-PKDD 2016, Riva del Garda, Italy, September 19, 2016, Revised Selected Papers",
% year="2017",
% publisher="Springer International Publishing",
% address="Cham",
% pages="19--35",
% isbn="978-3-319-61461-8",
% doi="10.1007/978-3-319-61461-8_2",
% url="https://doi.org/10.1007/978-3-319-61461-8_2"
% }
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV15, 
% author={G. Roffo and S. Melzi and M. Cristani}, 
% booktitle={2015 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Feature Selection}, 
% year={2015}, 
% pages={4202-4210}, 
% doi={10.1109/ICCV.2015.478}, 
% month={Dec}}
%  ------------------------------------------------------------------------
if nargin < 3
    alpha = 0.5;
end

if (alpha<0) || (alpha>1)
    fprintf( 'ERROR: 0 < alpha < 1 ...\n' );
    rankx = [];
    return;
end
    
% Shaode Yu
%    To make sure the input label is {-1, +1}
value = unique(Y_train);

if length(value) ~= 2
    fprintf( 'WARNING: ECFS.m is for binary classification ...\n' );
    rankx = [];
    return;
end

minY = min(value);
maxY = max(value);

Y_train( Y_train == minY ) = -1;
Y_train( Y_train == maxY ) = +1;

% ------------------------------------------------
% Preprocessing
s_n = X_train(Y_train==-1,:); %
s_p = X_train(Y_train==+1,:);
mu_sn = mean(s_n);
mu_sp = mean(s_p);

% Metric 1: Mutual Information
mi_s = [];
for i = 1:size(X_train,2)
    mi_s = [mi_s, muteinf(X_train(:,i),Y_train)];
end

% Metric 2: class separation
sep_scores = ([mu_sp - mu_sn].^2);
st   = std(s_p).^2;
st   = st+std(s_n).^2;
f=find(st==0); %% remove ones where nothing occurs
st(f)=10000;  %% remove ones where nothing occurs
sep_scores = sep_scores ./ st;

% Building the graph
vec = abs(sep_scores + mi_s )/2;

% Building the graph
Kernel_ij = [vec'*vec] ;

Kernel_ij = Kernel_ij - min(min( Kernel_ij ));
Kernel_ij = Kernel_ij./max(max( Kernel_ij ));

% Standard Deviation
STD = std(X_train,[],1);
STDMatrix = bsxfun( @max, STD, STD' );
STDMatrix = STDMatrix - min(min( STDMatrix ));
sigma_ij = STDMatrix./max(max( STDMatrix ));


Kernel =  (alpha*Kernel_ij+(1-alpha)*sigma_ij); 

% Eigenvector Centrality and Ranking
[eigVect, ~] = eigs(double(Kernel),1,'lm');
[~ , rankx ]= sort( abs(eigVect) , 'descend' );
rankx = rankx';
end

