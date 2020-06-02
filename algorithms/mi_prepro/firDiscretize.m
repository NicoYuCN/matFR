function [ data ] = firDiscretize( data, bins, xmethod )
% ---------------------------------------------------
% Return the given data array discretised into the specified number of
% bins of equal width. The returned array contains bin labels (not values).
% If the array is already discrete, it is not modified.
%   Input:
%       data, nSample * nFeature
%       bins, the number of bins for value discretization with equal width
%    xmethod, (1) equalValue, to keep interval values are equal (default)
%                    eg. max=1, min=0, bins=5, so binValue = (1-0)/5 = 0.2
%             (2) equalNumber, to keep the instances in each bin are equal
%                    eg, 100 samples, bins=5, so binNumber = 100/5 = 20
%  Output:
%       data, nSample * nFeature
% ---------------------------------------------------
if nargin < 2
    bins = 10;
end

if nargin < 3
    xmethod = 'equalValue';
end

if strcmp( 'equalValue', xmethod )
% this function keeps the interval values are equal in each bin
    % for instance, max=1, min=0, bins=5, so binValue = (1-0)/5 = 0.2
    data = equalValue( data, bins );
    
elseif strcmp( 'equalNumber', xmethod )
% this function keeps the interval number of instances are equal in each bin
    % for instance, there 100 samples, bins=5, so binNumber = 100/5 = 20
        % not so rationale
    data = equalNumber( data, bins );
    
else
    fprintf( 'WTF in firDiscretize.m (equalNumber or equalValue)? \n' );
    data = 0;
    return;
end
end

% --------------------------------------------------------------
function data = equalValue( data, bins )
% this function keeps the interval values are equal in each bin
%               eg. max=1, min=0, bins=5, so binValue = (1-0)/5 = 0.2

features = size( data, 2 );

for i = 1 : features
    values = data( :, i );
    
    % Discretize if not already discrete
    if length( unique( values ) ) > bins
        vmax = max( values );
        vmin = min( values );
        vbin = ( vmax - vmin ) / bins;
        
        for k = 1 : size( values )
            num = floor( ( values(k) - vmin ) / vbin );
            % ensure the end point of the interval is correctly mapped
            if num == bins
                num = bins - 1;
            end
            values( k ) = num;
        end
        
        data( :, i ) = values;
        
    end
end
end
% -----------------------------------------------

% -----------------------------------------------
function b = equalNumber( data, bins )
% this function keeps the number of instances are equal in each bin
%                    eg, 100 samples, bins=5, so binNumber = 100/5 = 20

[ n, dim ] = size( data );
b = zeros( n, dim );
for i = 1 : dim
    b( :, i ) = doDiscretize( data( :, i ), bins );
end
b = b + 1;
end

function y_discretized = doDiscretize( y, d )
% discretize a vector
ys = sort( y );
y_discretized = y;

pos = ys( round( length(y)/d * [1:d] ) );
for j = 1 : length(y)
    y_discretized( j ) = sum( y(j) > pos );
end
end

% -----------------------------------------------

