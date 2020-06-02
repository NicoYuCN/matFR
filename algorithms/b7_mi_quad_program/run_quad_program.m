function qp = run_quad_program( data, label )
% ----------------------------------------------------------------------
%  Support: Nguyen Xuan Vinh
%  E-mail: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com 
% Please refer to
%   Rodriguez-Lujan, I., Huerta, R., Elkan, C. and Cruz, C.S., 2010. 
%       Quadratic programming feature selection. Journal of Machine 
%           Learning Research, 11(Apr), pp.1491-1516.
% ----------------------------------------------------------------------
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%  Output:
%      qp     1*numF, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 2
    fprintf( 'Wrong: No enough inputs ( mifsQP.m ) ...\n' );
    qp = 0;
    return;
else
    
    fprintf( 'mifsQP starts ...... \n' );
    nFea = size( data, 2 );
    
    fprintf( '...... (1) compute MI matrix ......\n');
    H = computeMImatrix_4( [ data label ] );
    f = H( 1:end-1, end );
    H = H( 1:end-1, 1:end-1 );
    
    fprintf( '...... (2) eig-value analysis ......\n');
    evalue = eig( H );
    if evalue(1) < -10^-3
        fprintf('QPFS: non-positive Q\n');
        for i = 1:nFea 
            H( i, i ) = H( i, i ) + 3;
        end
    end
    
    fprintf( '...... (3) solving the QPFS formulation ......\n');
    mq = sum( sum(H) )/(nFea^2);
    mf = sum( f )/nFea;
    alpha = mq/( mq + mf );
    f = -alpha * f;
    H = ( 1 - alpha ) * H;
    
    A = -eye( nFea );  % x_i >=0 contraints
    b = zeros( nFea, 1 );
    Aeq = ones( 1, nFea );
    beq = 1;
    
    options = optimset( 'Algorithm', 'interior-point-convex' ); % embeded
    
    x = quadprog( H, f, A, b, Aeq, beq, [], [], [], options ); % embeded
    
    f = 0.5 * (1-alpha) * x' * H * x - alpha * f' * x;
    y = zeros( nFea, 2 );
    y( :, 1 ) = -x;
    for i = 1 : nFea 
        y( i, 2 ) = i;
    end
    
    y = sortrows( y );
    qp = y( :, 2 );
    qp = qp';    
    
    fprintf( 'run_quad_program ends here ...... \n' );
end
end