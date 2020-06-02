function cmim = rank_condition_mutual_info_max( data, label, numF )
% ----------------------------------------------------------------------
%  Support: Nguyen Xuan Vinh
%  E-mail: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
% Please refer to
%   Herman, G., Zhang, B., Wang, Y., Ye, G. and Chen, F., 2013. 
%       Mutual information-based method for selecting informative feature sets. 
%           Pattern Recognition, 46(12), pp.3315-3327.
% ----------------------------------------------------------------------
%   conditional mutual information maximization
% ---------------------------------------------------
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%      numF     k, the desired number of features ( k>=1 and k<=n )
%  Output:
%      cmim     1*numF, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 2
    fprintf( 'Wrong: No enough inputs ( mifsCMIM.m ) ...\n' );
    cmim = 0;
    return;
else
    if nargin < 3
        numF = size( data, 2 );
    end
    
    fprintf( 'mifsCMIM starts ...... \n' );
    
    nFea = size( data, 2 );
    
    fprintf( '...... (1) compute CMI matrix ......\n');
    H = computeCMImatrix_4( [ data label ] );
    H = H';
    
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
    
    fprintf( '...... (3) forward select of features based on MI ......\n');
    selected = zeros( 1, numF );
    selected( best_fs(1) ) = 1;
    
    for j = 2 : numF
        max_red = -inf; % max of min conditional relevancy
        bestFeature = 0;
        for i = 1 : nFea
            if selected(i) 
                continue;
            end
            
            mini = min( H( i, best_fs(1:j-1) ) );
            if max_red < mini
                max_red = mini;
                bestFeature = i;
            end
        end
        
        best_fs( j ) = bestFeature;
        selected( bestFeature ) = 1;
    end
    
    cmim = best_fs;
    
    fprintf( 'mifsCMIM ends here ...... \n' );
    
end
end
