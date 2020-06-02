function [ rankx, w] = rank_fir_concave_minimization_svm( X,Y,numF )
%  ------------------------------------------------------------------------
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
%  Please refer to
%   Bradley, P.S. and Mangasarian, O.L., 1998, July. Feature selection via 
%       concave minimization and support vector machines. 
%           In ICML (Vol. 98, pp. 82-90).
%  ------------------------------------------------------------------------
if nargin < 3
    numF = size(X,2);
end

%------------------------------------
value = unique(Y);
minY = min(value);
maxY = max(value);

Y( Y == minY ) = -1;
Y( Y == maxY ) = +1;
%-------------------------------------
loop=0;
finished=0;
alpha = 5; % Default
[m,n] = size(X);
v = zeros(n,1);



%% Main LOOP
while (~finished),
    loop=loop+1;    
    scale = alpha*exp(-alpha*v);   
    
    A=[diag(Y)*X, -diag(Y)*X, Y, -Y, -eye(m)];
    Obj = [scale',scale',0,0,zeros(1,m)];
    b = ones(m,1);
    x = slinearsolve(Obj',A,b,Inf);
    w = x(1:n)-x(n+1:2*n);
    b0 = x(2*n+1)-x(2*n+2);
    vnew=abs(w);
    
    if (norm(vnew-v,1)<10^(-5)*norm(v,1)),
        finished=1;
    else
        v=vnew;
    end;
    if (loop>2),
        finished=1;
    end;
    nfeat=length(find(vnew>100*eps));
    
    disp(['Iter ' num2str(loop) ' - feat ' num2str(nfeat)]);
    
    if nfeat<numF,
        finished=1;
    end;
end;

[~, ind] = sort(-abs(w));

rankx=ind';


end

