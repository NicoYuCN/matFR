function [prob] = probs(feat)
% Compute probability density of binned feature vector.
% Returns vector of probabilities, for bins 0,1,...

bins=max(feat)+1;
prob=zeros(1,bins);

for k=1:length(feat)
    f_cur=feat(k)+1;
    prob(f_cur) = prob(f_cur)+1;
end

prob = prob ./ sum(prob);
