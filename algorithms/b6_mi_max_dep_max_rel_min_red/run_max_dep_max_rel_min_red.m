function mrmr = run_max_dep_max_rel_min_red( data, label, numF )
% ----------------------------------------------------------------------
%  Support: Nguyen Xuan Vinh
%  E-mail: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
% Please refer to
%   Peng, H., Long, F. and Ding, C., 2005. Feature selection based on 
%       mutual information: Criteria of max-dependency, max-relevance, and 
%           min-redundancy. IEEE Transactions on Pattern Analysis & Machine
%               Intelligence, (8), pp.1226-1238.
% ----------------------------------------------------------------------
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%      numF     k, the desired number of features ( k>=1 and k<=n )
%  Output:
%      mrmr     1*numF, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 2
    fprintf( 'Wrong: No enough inputs ( mrmrMIFS.m ) ...\n' );
    mrmr = 0;
    return;
else
    if nargin < 3
        numF = size( data, 2 );
    end
    
    fprintf( 'mrmrMIFS starts ...... \n' );
    
    nFea = size( data, 2 );
    
    fprintf( '...... (1) compute MI matrix ......\n');
    H = computeMImatrix_4( [ data label ] );
    f = H( 1:end-1, end );
    H = H( 1:end-1, 1:end-1 );
    
    fprintf( '...... (2) select the 1st feature with max MI ......\n');
    max_MI = 0;
    firstFeature = 1;
    for i = 1 : nFea
        CMI = f( i );
        if CMI > max_MI
            max_MI = CMI;
            firstFeature = i;
        end
    end
    
    best_fs = zeros( 1, numF );
    best_fs(1) = firstFeature;
    
    fprintf( '...... (3) forward select of features based on MI ......\n');
    selected = zeros( 1, numF );
    selected( best_fs(1) ) = 1;
    for j = 2 : numF
        max_inc = -inf;
        bestFeature = 0;
        for i = 1 : nFea
            if selected( i )
                continue;
            end
            
            rel = f( i );
            red = sum( H( i, best_fs( 1:j-1 ) ) )/( j-1 );
            
            inc = rel - red;
            if inc > max_inc
                max_inc = inc;
                bestFeature = i;
            end
        end
        
        best_fs( j ) = bestFeature;
        selected( bestFeature ) = 1;
    end
    
    mrmr = best_fs;
    
    fprintf( 'run_max_dep_max_rel_min_red ends here ...... \n' );
end
end