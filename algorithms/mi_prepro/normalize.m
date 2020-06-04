function n = normalize(X)
% Normalize X so that each column has mean 0 and standard deviation 1.

m=mean(X);
s=std(X);
for i=1:size(X,1)
    X(i,:)=(X(i,:)-m)./s;
end

n=X;
