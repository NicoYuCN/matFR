function [ r ] = matFR_fn( X, f, Y, m4norm )
% -------------------------------------------------------------------------
% Zhicheng Zhang, Xiaokun Liang, Shaode Yu, Yaoqin Xie
% May 25, 2020
% Email: yushaodemia@163.com
%   Dedicated to methods excluding mutual information (MI) methods
% -------------------------------------------------------------------------
% if the matFR toolbox is useful, please refer to
%     Zhang Z, Liang X, Qin W, Yu S, Xie Y. matFR: a MATLAB toolbox for 
%               feature ranking. Bioinformatics. 2020 Oct 1;36(19):4968-9.
% -------------------------------------------------------------------------
% Inputs
%       X,      a matrix shows samples and their features
%                   its size [m, n] indicates m samples and n featuers per sample
%       f,      a feature selection method
%       Y,      the corresponding labels to each samples
%                   its size [m, 1] and the Y values are in {0, 1}
%       m4norm, a method for data normalization
%                   (a) 'zscore':  mean=0; std=1
%                   (b) 'normc':   sum( c_1 .* c_1 ) = 1
%                   (c) 'linear':  (x-min (x) )/(max(x) - min(x) )
%                   (d) 'clinear':  x/max(Aabs(x))
%                   (e) no normalization (default)
% Output
%       r,      the feature ranks from the most to the least importance
%                   its size [1, n]
% -------------------------------------------------------------------------
% (0) to check the input parameters
if nargin == 4
    X = firDataNorm( X, m4norm );
end

if nargin == 3
    if size(X,1) ~= size(Y,1)
        fprintf('Error: Input X and Y not matched ...\n');
        r = [];
        return;
    end
end

if nargin == 1
    f = 'd1_mat_ttest'; % a default method
end

% -------------------------------------------------------------------------
% (1) to use a method for feature importance ranking
selection_method = f;
switch (selection_method)
    % (1.01)
    case 'd1_mat_ttest'
        fprintf('...(d1_mat_ttest_1987)...\n')
        fprintf( '===*** 1987. Guinness, Gosset, Fisher, and small samples ...===\n' );
        r = rank_mat_ttest( X, Y );
        fprintf('...................................................................\n');
    % (1.02)
    case 'd2_mat_entropy'
        fprintf('...(d2_mat_entropy_1951)...\n')
        fprintf( '===*** 1951. On information and sufficiency ...===\n' );
        r = rank_mat_entropy( X, Y );
        fprintf('...................................................................\n');
    % (1.03)
    case 'd3_mat_bhattacharyya'
        fprintf('...(d3_mat_bhattacharyya_1952)...\n')
        fprintf( '===*** 1952. A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations ...===\n' );
        r = rank_mat_bhattacharyya( X, Y );
        fprintf('...................................................................\n');
    % (1.04)
    case 'd4_mat_roc'
        fprintf('...(d4_mat_roc_1997)...\n')
        fprintf( '===*** 1997. The use of the area under the ROC curve in the evaluation of machine learning algorithms ...===\n' );
        r = rank_mat_roc( X, Y );
        fprintf('...................................................................\n');
    % (1.05)
    case 'd5_mat_wilcoxon'
        fprintf('...(d5_mat_wilcoxon_1992)...\n')
        fprintf( '===*** 1992. Individual comparisons by ranking methods ...===\n' );
        r = rank_mat_wilcoxon( X, Y );
        fprintf('...................................................................\n');
    % (1.06)
    case 'd6_mat_relieff'
        fprintf('...(d6_mat_relieff_1997)...\n')
        fprintf( '===*** 1997. Overcoming the myopia of inductive learning algorithms with RELIEFF ...===\n' );
        r = rank_mat_relieff( X, Y );
        fprintf('...................................................................\n');
    % (1.07)
    case 'd7_mat_lasso'
        fprintf('...(d7_mat_lasso_1996)...\n')
        fprintf( '===*** 1996. Regression shrinkage and selection via the lasso ...===\n' );
        r = rank_mat_lasso( X, Y );
        fprintf('...................................................................\n');
    % (1.08)
    case 'e1_fir_correlation'
        fprintf('...(e1_fir_correlation_2016)...\n')
        fprintf( '===*** 2016. Feature selection library (MATLAB toolbox) ...===\n' );
        r = rank_fir_correlation( X );
        fprintf('...................................................................\n');
    % (1.09)
    case 'e2_fir_fisher'
        fprintf('...(e2_fir_fisher_2012)...\n')
        fprintf( '===*** 2012. Generalized fisher score for feature selection ...===\n' );
        r = rank_fir_fisher( X, Y );
        fprintf('...................................................................\n');
    % (1.10)
    case 'e3_fir_gini'
        fprintf('...(e3_fir_gini_1997)...\n')
        fprintf( '===*** 1997. Concentration and dependency ratios ...===\n' );
        r = rank_fir_gini( X, Y );
        fprintf('...................................................................\n');
    % (1.11)
    case 'e4_fir_KruskalWallis'
        fprintf('...(e4_fir_KruskalWallis_2013)...\n')
        fprintf( '===*** 2013. Nonparametric statistical methods ...===\n' );
        r = rank_fir_KruskalWallis( X, Y );
        fprintf('...................................................................\n');
    % (1.12)
    case 'g1_fir_pair_wise_feature_proximity'
        fprintf('...(g1_fir_pair_wise_feature_proximity_2017)...\n')
        fprintf( '===*** 2017. An effective feature selection method based on pair-wise feature proximity for high dimensional low sample size data ...===\n' );
        r = rank_pair_wise_feature_proximity( X, Y );
        fprintf('...................................................................\n');
    % (1.13)
    case 'g2_fir_max_min_local_structure_info'
        fprintf('...(g2_fir_max_min_local_structure_info_2013)...\n')
        fprintf( '===*** 2013. Minimum¨Cmaximum local structure information for feature selection ...===\n' );
        r = rank_fir_max_min_local_structure_info( X, Y );
        fprintf('...................................................................\n');
    % (1.14)
    case 'g3_fir_local_learning_clustering'
        fprintf('...(g3_fir_local_learning_clustering_2010)...\n')
        fprintf( '===*** 2010. Feature selection and kernel learning for local learning-based clustering ...===\n' );
        r = rank_fir_local_learning_clustering( X );
        fprintf('...................................................................\n');
    % (1.15)
    case 'g4_fir_L12_regu_discrime'
        fprintf('...(g4_fir_L12_regu_discrime_2011)...\n')
        fprintf( '===*** 2011. L2,1-norm regularized discriminative feature selection for unsupervised learning ...===\n' );
        r = rank_fir_L12_regu_discrime( X );
        fprintf('...................................................................\n');
    % (1.16)
    case 'g5_fir_eigenvector_centrality'
        fprintf('...(g5_fir_eigenvector_centrality)...\n')
        fprintf( '===*** 2016. Features Selection via Eigenvector Centrality ...===\n' );
        r = rank_fir_eigenvector_centrality( X, Y );
        fprintf('...................................................................\n');
    % (1.17)
    case 'g6_fir_infinite_latent'
        fprintf('...(g6_fir_infinite_latent_2017)...\n')
        fprintf( '===*** 2017. Infinite latent feature selection: A probabilistic latent graph-based ranking approach ...===\n' );
        r = rank_fir_infinite_latent( X, Y );
        fprintf('...................................................................\n');
    % (1.18)
    case 'g7_fir_concave_minimization'
        fprintf('...(g7_fir_concave_minimization_1998)...\n')
        fprintf( '===*** 1998. Feature selection via concave minimization and support vector machines ...===\n' );
        r = rank_fir_concave_minimization_svm( X, Y );
        fprintf('...................................................................\n');
    % (1.19)
    case 'g8_fir_infinite'
        fprintf('...(g8_fir_infinite_2015)...\n')
        fprintf( '===*** 2015. Infinite feature selection ...===\n' );
        r = rank_fir_infinite( X, Y );
        fprintf('...................................................................\n');
    % (1.20)
    case 'g9_fir_ordinal_locality'
        fprintf('...(g9_fir_ordinal_locality_2017)...\n')
        fprintf( '===*** 2017. Unsupervised feature selection with ordinal locality ...===\n' );
        r = rank_fir_ordinal_locality( X );
        fprintf('...................................................................\n');
    % (1.21)
    case 'h1_fir_structured_graph_optimization'
        fprintf('...(h1_fir_structured_graph_optimization_2016)...\n')
        fprintf( '===*** 2016. Unsupervised feature selection with structured graph optimization ...===\n' );
        r = rank_structured_graph_optimization( X );
        fprintf('...................................................................\n');
    % (1.22)
    case 'h2_fir_laplacian_score'
        fprintf('...(h2_fir_laplacian_score_2005)...\n')
        fprintf( '===*** 2005. Laplacian score for feature selection ...===\n' );
        r = rank_fir_laplacian_score( X );
        fprintf('...................................................................\n');
    % (1.23)
    case 'h3_fir_simul_ortho_clustering'
        fprintf('...(h3_fir_simul_ortho_clustering_2015)...\n')
        fprintf( '===*** 2015. Unsupervised simultaneous orthogonal basis clustering feature selection ...===\n' );
        r = rank_fir_simul_ortho_clustering( X );
        fprintf('...................................................................\n');
    % (1.24)
    case 'h4_fir_multi_cluster'
        fprintf('...(h4_fir_multi_cluster_2010)...\n')
        fprintf( '===*** 2010. Unsupervised feature selection for multi-cluster data ...===\n' );
        r = rank_fir_multi_cluster( X );
        fprintf('...................................................................\n');
    % (1.25)
    case 'h5_fir_dependence_guided'
        fprintf('...(h5_fir_dependence_guided_2018)...\n')
        fprintf( '===*** 2018. Dependence guided unsupervised feature selection ...===\n' );
        r = rank_dependence_guided( X );
        fprintf('...................................................................\n');
    % (1.26)
    case 'h6_fir_adaptive_structure_learning'
        fprintf('...(h6_fir_adaptive_structure_learning_2015)...\n')
        fprintf( '===*** 2015. Unsupervised feature selection with adaptive structure learning ...===\n' );
        r = rank_fir_adaptive_structure_learning( X );
        fprintf('...................................................................\n');
    % (1.27)
    case 'k1_fir_joint_embed_learn_sparse_regression'
        fprintf('...(k1_fir_joint_embed_learn_sparse_regression_2011)...\n')
        fprintf( '===*** 2011. Feature selection via joint embedding learning and sparse regression ...===\n' );
        r = rank_joint_embed_learn_sparse_regression( X );
        fprintf('...................................................................\n');
    % (1.28)
    case 'k2_fir_spectrum_info_graph_laplacian'
        fprintf('...(k2_fir_spectrum_info_graph_laplacian_2007)...\n')
        fprintf( '===*** 2007. Spectral feature selection for supervised and unsupervised learning feature selection based on spectral information of graph laplacian ...===\n' );
        r = rank_spectrum_info_graph_laplacian( X );
        fprintf('...................................................................\n');
    % (1.29)
    case 'k3_fir_nonneg_spectral_analysis'
        fprintf('...(k3_fir_nonneg_spectral_analysis_2012)...\n')
        fprintf( '===*** 2012. Unsupervised feature selection using nonnegative spectral analysis ...===\n' );
        r = rank_fir_nonneg_spectral_analysis( X );
        fprintf('...................................................................\n');
    % (1.30)
    case 'k4_fir_robust_unsupervised'
        fprintf('...(k4_fir_robust_unsupervised_2013)...\n')
        fprintf( '===*** 2013. Robust unsupervised feature selection ...===\n' );
        r = rank_fir_robust_unsupervised( X );
        fprintf('...................................................................\n');
    % (1.31)
    case 'm1_fir_self_growing_forest' 
        fprintf('...(m1_fir_self_growing_forest)...\n')
        fprintf( '===*** 2022. Robust unsupervised feature selection ...===\n' );
        r = rank_self_growing_forest( X, Y );
        fprintf('...................................................................\n');
    %
    otherwise
        fprintf( 'Please check the method \n' );
        r = [];
        fprintf('...................................................................\n');
end

end

