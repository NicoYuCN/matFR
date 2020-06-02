function miq = run_min_redundancy( data, label, numF )
% ----------------------------------------------------------------------
%  Support: Nguyen Xuan Vinh
%  E-mail: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
% Please refer to
%   Ding, C. and Peng, H., 2005. Minimum redundancy feature selection from 
%       microarray gene expression data. Journal of bioinformatics and 
%           computational biology, 3(02), pp.185-205.
% ----------------------------------------------------------------------
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%      numF     k, the desired number of features ( k>=1 and k<=n )
%  Output:
%      mrmr     1*numF, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 2
    fprintf( 'Wrong: No enough inputs ( run_min_redundancy.m ) ...\n' );
    miq = 0;
    return;
else
    if nargin < 3
        numF = size( data, 2 );
    end
    
    fprintf( 'mifsMIQ starts ...... \n' );
    
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
    selected = zeros( 1, nFea );
    selected( best_fs(1) ) = 1;

    for j = 2 : numF
        max_inc = -inf;
        bestFeature = 0;
        for i = 1 : nFea
            if selected(i) 
                continue;
            end
            
            rel = f(i);
            red = sum( H( i, best_fs(1:j-1) ) )/(j-1);
            
            inc = rel/red;  % miq criterion
            if inc > max_inc
                max_inc = inc;
                bestFeature = i;
            end
        end
        
        best_fs(j) = bestFeature;
        selected( bestFeature ) = 1;
    end
    
    miq = best_fs;
    
    fprintf( 'run_min_redundancy ends here ...... \n' );
    
end
end



