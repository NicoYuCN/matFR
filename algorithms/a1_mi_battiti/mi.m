function [I] = mi(feat1,feat2)
% Mutual information between discrete (or binned) features

if max(feat1)>length(feat1)
    omax=max(feat1);
    feat1=ordering(feat1);
end
if max(feat2)>length(feat2)
    omax=max(feat2);
    feat2=ordering(feat2);
end

bins1=max(feat1)+1;
bins2=max(feat2)+1;

p1=probs(feat1);
p2=probs(feat2);
jp=jointprobs(feat1,feat2);

I=0;
for i=1:bins1
    for j=1:bins2
        pr=jp(i,j);
        o1=p1(i);
        o2=p2(j);

        if (pr~=0 && o1~=0 && o2~=0)
            I = I + pr * log2(pr/(o1*o2));
        end
    end
end

