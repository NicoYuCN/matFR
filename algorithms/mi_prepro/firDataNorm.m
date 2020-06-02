function [ dataNorm ] = firDataNorm( data, method )
% ---------------------------------------------------
% Shaode Yu, 05/10/2019, nicolasyude@163.com
%   methods for normalization
%             (a)  'zscore':  mean=0; std=1  (default)
%             (b)   'normc':  sum( c_1 .* c_1 ) = 1
%             (c)  'linear':  (x-min (X) )/(max(X) - min(X) )
%             (d) 'clinear':  x/max(abs(X))
%   while in Mutial Information analysis, (c) is equal to (d)
% ---------------------------------------------------
%   Input:
%         data, nSample * nFeature
%       method, 'zscore', 'normc', 'linear', 'clinear'
%  Output:
%     dataNorm, data after normalization
% ---------------------------------------------------
if nargin < 2
    method = 'zscore';
end

[ numSample, numFeature ] = size( data );

switch lower(method)
    case 'zscore'
        dataNorm = normalize( data );
    case 'normc'
        dataNorm = normc( data );
    case 'linear'
        dataNorm = zeros( numSample, numFeature );
        for ii = 1:numFeature
            tmpFeature = data( :, ii );
            tmpMin = min( tmpFeature );
            tmpMax = max( tmpFeature );
            dataNorm(:,ii) = ( tmpFeature - tmpMin ) / ( tmpMax - tmpMin );
        end
        
    case 'clinear' % coarse linear mapping - equal to linear in MI
        dataNorm = zeros( numSample, numFeature );
        for ii = 1:numFeature
            tmpFeature = data( :, ii );
            tmpMax = max( abs(tmpFeature) );
            dataNorm(:,ii) = ( tmpFeature ) / ( tmpMax );
        end
    otherwise
        disp('Unknown method for data normalization and EMPTY returned. \n');
        dataNorm = [];
end
end

