function specCMI = run_joint_mutual_info( data, label )
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
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%  Output:
%      specCMI  1*numF, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 2
    fprintf( 'Wrong: No enough inputs ( mifsSPECcmi.m ) ...\n' );
    specCMI = 0;
    return;
else
    
    fprintf( 'mifsCMIM starts ...... \n' );
    
    nFea = size( data, 2 );
    
    fprintf( '...... (1) compute CMI matrix ......\n');
    H = computeCMImatrix_4( [ data label ] );
    
    fprintf( '...... (2) eig-value analysis ......\n');    
    H = ( H + H' )/2;
    [ V, ~ ] = eigs( H, 1 );
    
    fprintf( '...... (3) forward select of features ......\n'); 
    x = V( :, 1 );
    x = abs( x/norm(x) );
    y = zeros( nFea, 2 );
    y( :, 1 ) = -x;

    for i = 1 : nFea
        y( i, 2 ) = i;
    end
    
    y = sortrows( y );
    specCMI = y( :, 2 )';
    
    fprintf( 'mifsSPECcmi ends here ...... \n' );
    
end
end
