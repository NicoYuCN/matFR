function rankx = rank_step_wise(X,T)
% ----------------------------------------------------------------------
% Please refer to
%   Cang, S. and Partridge, D., 2004. 
%       Feature ranking and best feature subset using mutual information. 
%           Neural Computing & Applications, 13(3), pp.175-184.
% ----------------------------------------------------------------------
% step-wise mutual information based feature ranking
% ----------------------------------------------------------------------

% Find number of features.
nf=size(X,2);

% Initialize F as the full set of features and S as the empty set
% (sets are encoded as arrays of indices to the data matrix).
F=1:nf;
S=[];

% Handle representation of classes by mapping to an integer in the
% interval [1,classes].
classes=size(T,2);
cl=zeros(size(T,1),1);
for i=1:size(T,1)
    for k=1:classes
        if T(i,k)==1
            cl(i)=k;
            break;
        end
    end
end

% Here's the greedy feature ranking part.
currentset=[];
while length(F)>0
    % Init with an unrealistically small number.
    max_magic=-nf;
    max_idx=0;

    % Check each f in F.
    for p=1:length(F)
        f=F(p);

        % compute I(S,f;Z)
        magic=mi(composite(currentset,X(:,f)),cl);
        if magic>max_magic
            max_magic=magic;
            max_idx=p;
        end
    end
    
    % Move next chosen feature from S to F.
    fprintf('Choice #%d: feature #%d, bringing total composite MI to %f\n', length(S)+1,F(max_idx),max_magic);
    S(length(S)+1)=F(max_idx);
    F(max_idx)=[];
    % Update the "current set"
    currentset=ordering(composite(currentset,X(:,S(length(S)))));
end

% Print the results for debugging purposes.
%fprintf(1,'Resulting order of features: ');
%for i=1:nf
%    for k=1:length(S)
%        if S(k)==i
%            fprintf(1,'%d  ',k);
%        end
%    end
%end

% Return ranking of features. 
% The most important feature is ranking first.
rankx=S;
