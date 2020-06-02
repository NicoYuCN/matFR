function [prob] = jointprobs(feat1,feat2)
% Compute joint probability density of binned feature vectors. Vectors are
% assumed to have the same length.
% Returns matrix of probabilities, where element m(i,j) is joint
% probability for feat1 bin i and feat2 bin j.

bins1=max(feat1)+1;
bins2=max(feat2)+1;
prob=zeros(bins1,bins2);

for k=1:length(feat1)
    f1=feat1(k)+1;
    f2=feat2(k)+1;
    prob(f1,f2) = prob(f1,f2)+1;
end

prob = prob ./ sum(sum(prob));
