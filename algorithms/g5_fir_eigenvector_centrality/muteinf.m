function info = muteinf(A, Y)
n = size(A,1);
Z = [A Y];
if(n/10 > 20)
    nbins = 20;
else
    nbins = max(floor(n/10),10);
end;
pA = hist(A, nbins);
pA = pA ./ n;

i = find(pA == 0);
pA(i) = 0.00001;

od = size(Y,2);
cl = od;
if(od == 1)
    pY = [length(find(Y==+1)) length(find(Y==-1))] / n;
    cl = 2;
else
    pY = zeros(1,od);
    for i=1:od
        pY(i) = length(find(Y==+1));
    end;
    pY = pY / n;
end;
p = zeros(cl,nbins);
rx = abs(max(A) - min(A)) / nbins;
for i = 1:cl
    xl = min(A);
    for j = 1:nbins
        if(i == 2) && (od == 1)
            interval = (xl <= Z(:,1)) & (Z(:,2) == -1);
        else
            interval = (xl <= Z(:,1)) & (Z(:,i+1) == +1);
        end;
        if(j < nbins)
            interval = interval & (Z(:,1) < xl + rx);
        end;
        %find(interval)
        p(i,j) = length(find(interval));
        
        if p(i,j) == 0 % hack!
            p(i,j) = 0.00001;
        end
        
        xl = xl + rx;
    end;
end;
HA = -sum(pA .* log(pA));
HY = -sum(pY .* log(pY));
pA = repmat(pA,cl,1);
pY = repmat(pY',1,nbins);
p = p ./ n;
info = sum(sum(p .* log(p ./ (pA .* pY))));
info = 2 * info ./ (HA + HY);