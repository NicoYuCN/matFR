function u = ordering(v)
% Return ordering of vector components, i.e. lowest value mapped to 1,
% second lowest to 2, etc.
%
% Postconditions: If two elements differ in v, the corresponding elements
% in u also differ. For any two elements in v that are equal, the
% corresponding elements in u are equal.

% array of indices not mapped yet
idx=1:length(v);
count=1;

while length(idx)>0
    % find the lowest element in remaining v
    minval=min(v(idx));

    % find occurrences of minimum value, set the corresponding items in u
    % and remove indices from index list (the downward loop is necessary)
    for q=length(idx):-1:1
        i=idx(q);
        if v(i)==minval
            u(i)=count;
            idx(q)=[];
        end
    end
    count=count+1;
end
