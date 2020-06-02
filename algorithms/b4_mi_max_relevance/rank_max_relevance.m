function maxrel = rank_max_relevance( data, label )
% ----------------------------------------------------------------------
%  Support: Nguyen Xuan Vinh
%  E-mail: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
% Please refer to
%   Battiti, R., 1994. Using mutual information for selecting features in 
%       supervised neural net learning. IEEE Transactions on neural 
%           networks, 5(4), pp.537-550.
% ----------------------------------------------------------------------
% a greedy feature selection algorithm based on mutual information
% ----------------------------------------------------------------------
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%  Output:
%      mrmr     1*numF, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 2
    fprintf( 'Wrong: No enough inputs ( mifsMaxRel.m ) ...\n' );
    maxrel = 0;
    return;
else
    
    fprintf( 'mifsQP starts ...... \n' );
    nFea = size( data, 2 );
    
    fprintf( '...... (1) compute MI matrix ......\n');
    H = computeMImatrix_4( [ data label ] );
    f = H( 1:end-1, end );
    H = H( 1:end-1, 1:end-1 );
    
    fprintf( '...... (2) sort features ......\n');
    y = zeros( nFea, 2 );
    for i = 1:nFea
        y( i, 1 ) = -f(i);
        y( i, 2 ) = i;
    end
    
    y = sortrows( y );
    maxrel = y( :, 2 )';
    
    fprintf( 'rank_max_relevance ends here ...... \n' );
    
end
end