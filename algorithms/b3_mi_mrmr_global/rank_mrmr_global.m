function mrmtr = rank_mrmr_global( data, label, numF )
% ----------------------------------------------------------------------
%  Support: Nguyen Xuan Vinh
%  E-mail: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
% Please refer to
%   Nguyen, X.V., Chan, J., Romano, S. and Bailey, J., 2014, August. 
%       Effective global approaches for mutual information based feature 
%           selection. In Proceedings of the 20th ACM SIGKDD international 
%               conference on Knowledge discovery and data mining 
%                   (pp. 512-521). ACM.
% ----------------------------------------------------------------------
% 
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%      numF     k, the desired number of features ( k>=1 and k<=n )
%  Output:
%      mrmtr    1*numF, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 2
    fprintf( 'Wrong: No enough inputs ( mifsMRMTR.m ) ...\n' );
    mrmtr = 0;
    return;
else
    if nargin < 3
        numF = size( data, 2 );
    end
    
    fprintf( 'mifsMRMTR starts ...... \n' );
    
    nFea = size( data, 2 );
    
    fprintf( '...... (1) compute CMI matrix ......\n');
    H = computeCMImatrix_4( [ data label ] );
    
    fprintf( '...... (2) select the 1st feature with max MI ......\n');
    max_MI = 0;
    firstFeature = 1;
    for i = 1 : nFea
        CMI = H( i, i );
        if CMI > max_MI
            max_MI = CMI;
            firstFeature = i;
        end
    end
    
    best_fs = zeros( 1, numF );
    best_fs(1) = firstFeature;
    
    fprintf( '...... (2) create the JMI matrix ......\n');
    for i = 1 : nFea
        for j = 1 : nFea
            if j == i 
                continue;
            end
            
            H( i, j ) = H( i, i ) + H( i, j );  %I(Xi;C) + I(Xj;C|Xi)= I(Xi,Xj;C)
        end
    end
    
    fprintf( '...... (3) forward select of features based on MI ......\n');
    selected = zeros( 1, nFea );
    selected( best_fs(1) ) = 1;
    for j = 2 : numF
        max_inc = -inf;
        bestFeature = 0;
        for i = 1 : nFea
            if selected(i) 
                continue;
            end
            
            totalJMI = sum( H( i, best_fs(1:j-1) ) );
            if totalJMI > max_inc
                max_inc = totalJMI;
                bestFeature = i;
            end
        end
        
        best_fs(j) = bestFeature;
        selected( bestFeature ) = 1;
    end
    
    mrmtr = best_fs;
    
    fprintf( 'mifsMRMTR ends here ...... \n' );
    
end
end

