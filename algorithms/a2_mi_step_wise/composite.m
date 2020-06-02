function [cf] = composite(feat1,feat2,varargin)
% Create a composite feature from two or more discrete, binned features.
% Feature vectors should have equal lengths.
% Warning: composing a large number of features results in very wide value
% ranges.

% handle empty lists as arguments, and more than 2 arguments
if nargin > 2
    for n=1:nargin-2
        q=varargin{n}
        feat2=m_composite(feat2,varargin{n})
    end
elseif length(feat1)==0
    cf=feat2;
    return;
elseif length(feat2)==0
    cf=feat1;
    return;
end

bins1=max(feat1)+1;
bins2=max(feat2)+1;

cf=zeros(size(feat1));
mult=max(bins1,bins2);

for i=1:length(feat1)
    cf(i) = mult*feat1(i) + feat2(i);
end
