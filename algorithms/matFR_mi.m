function [ r ] = matFR_mi( X, f, Y, m4norm, nbin )
% -------------------------------------------------------------------------
% Zhicheng Zhang, Xiaokun Liang, Shaode Yu, Yaoqin Xie
% May 25, 2020
% Email: yushaodemia@163.com
%   Dedicated to mutual information (MI) based methods
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
%                   (a) 'zscore':  mean=0; std=1 (default)
%                   (b) 'normc':   sum( c_1 .* c_1 ) = 1
%                   (c) 'linear':  (x-min (x) )/(max(x) - min(x) )
%                   (d) 'clinear':  x/max(Aabs(x))
%       nbin,   the number of bins
%
% Output
%       r,      the feature ranks from the most to the least importance
%                   its size [1, n]
% -------------------------------------------------------------------------
% (0) to check the input parameters
if nargin < 5
    nbin = 8; % firDiscretize.m
end

if nargin < 4
    m4norm = 'zscore'; % firDataNorm.m
end

if nargin == 3 % a supervised method
    if size(X,1) ~= size(Y,1)
        fprintf('Error: Input X and Y not matched ...\n');
        r = [];
        return;
    end
end

if nargin == 2
    f = 'b6_mi_max_dep_max_rel_min_red';
end

% -------------------------------------------------------------------------
% (1) to prepare the data
% (1.a) feature normalization
X = firDataNorm( X, m4norm );
% (1.b) data discretization
X = firDiscretize( X, nbin );

% -------------------------------------------------------------------------
% (2) mi-based feature ranking
selection_method = f;
switch (selection_method)
    % (3.01)
    case 'a1_mi_battiti'
        fprintf('...(a1_mi_battiti_1994)...\n')
        fprintf( '===*** 1994. Using mutual information for selecting features in supervised neural net learning ...===\n' );
        r = rank_battiti( X, Y );
        fprintf('...................................................................\n');
    % (3.02)
    case 'a2_mi_step_wise'
        fprintf('...(a2_mi_step_wise_2004)...\n')
        fprintf( '===*** 2004. Feature ranking and best feature subset using mutual information ...===\n' );
        r = rank_step_wise( X, Y );
        fprintf('...................................................................\n');
    % (3.03)
    case 'b1_mi_cond_infomax_learn'
        fprintf('...(b1_mi_cond_infomax_learn_2006)...\n')
        fprintf( '===*** 2006. Conditional infomax learning: An integrated framework for feature extraction and fusion ...===\n' );
        r = rank_condition_infomax_learning( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.04)
    case 'b2_mi_cond_mutual_info_max'
        fprintf('...(b2_mi_cond_mutual_info_max_2013)...\n')
        fprintf( '===*** 2013. Mutual information-based method for selecting informative feature sets ...===\n' );
        r = rank_condition_mutual_info_max( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.05)
    case 'b3_mi_mrmr_global'
        fprintf('...(b3_mi_mrmr_global_2014)...\n')
        fprintf( '===*** 2014. Effective global approaches for mutual information based feature selection ...===\n' );
        r = rank_mrmr_global( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.06)
    case 'b4_mi_max_relevance'
        fprintf('...(b4_mi_max_relevance_battiti_1994)...\n')
        fprintf( '===*** 1994. Using mutual information for selecting features in supervised neural net learning ...===\n' );
        r = rank_max_relevance( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.07)
    case 'b5_mi_min_redundancy'
        fprintf('...(b5_mi_min_reduandancy_2014)...\n')
        fprintf( '===*** 2014. Effective global approaches for mutual information based feature selection ...===\n' );
        r = rank_mi_min_redundancy( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.08)
    case 'b6_mi_max_dep_max_rel_min_red'
        fprintf('...(b6_mi_max_dep_max_rel_min_red_2005)...\n')
        fprintf( '===*** 2005. Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy ...===\n' );
        r = run_max_dep_max_rel_min_red( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.09)
    case 'b7_mi_quad_program'
        fprintf('...(b7_mi_quad_program_2010)...\n')
        fprintf( '===*** 2010. Quadratic programming feature selection ...===\n' );
        r = run_quad_program( X, Y );
        fprintf('...................................................................\n');
    % (3.10)
    case 'b8_mi_min_redundancy'
        fprintf('...(b8_mi_min_redundancy_2005)...\n')
        fprintf( '===*** 2005. Minimum redundancy feature selection from microarray gene expression data ...===\n' );
        r = run_min_redundancy( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.11)
    case 'b9_mi_joint'
        fprintf('...(b9_mi_joint_2014)...\n')
        fprintf( '===*** 2014. Effective global approaches for mutual information based feature selection ...===\n' );
        r = run_joint_mutual_info( X, Y+1 );
        fprintf('...................................................................\n');
    % (3.12)
    case 'c1_mi_giorgio'
        fprintf('...(c1_mi_giorgio_2018)...\n')
        fprintf( '===*** 2018. Feature Selection Library (MATLAB Toolbox) ...===\n' );
        r = rank_mi_giorgio( X, Y );
        fprintf('...................................................................\n');
    % (3.13)
    otherwise
        fprintf( 'Please check the method \n' );
        r = [];
        fprintf('...................................................................\n');
end



end

