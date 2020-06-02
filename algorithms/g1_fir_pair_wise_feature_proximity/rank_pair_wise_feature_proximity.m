function rankx = rank_pair_wise_feature_proximity(X,Y,percDim)
% ---------------------------------------------------------------------------------
% author : S L Happy
% input:  X - data (nxd) [n samples and d dimensions]
%             each row of X is an observation and each column is a variable
%         Y - labels of data X. map labels into integers.
%             dimension (nx1)
%	  percDim - percentage of dimensions to be used for keeping the 
%	      realated pair-wise features (scalar)
% output: ranking: ranking of each feature 
%
% Example:
% 
%     X = rand(30,20);
%     Y = [ones(10,1),  2*ones(10,1),   3*ones(10,1)];
%     ranking1 = pwfp(X,Y);
%     percDim = 12;    % recommended value is between 8 to 20 for HDLSS
%     ranking2 = pwfp(X,Y,percDim);
%     selectedFeatures = ranking2(1:5);   % select the best 5 features
%
% Note, If you use our code or method, please cite our paper:
%       S L Happy, R. Mohanty, and A. Routray, "An effective feature selection 
%	method based on pair-wise feature proximity for high dimensional 
%	low sample size data," in European Signal Processing Conf. (EUSIPCO), 2017.
% ---------------------------------------------------------------------------------

if nargin<2
    error('the class labels are required!')
end
if nargin<3
    percDim = 10;   % default vcalue of percDim is 10.
end
[~,~,Y] = unique(Y);
percDim = percDim/100;
dim = size(X,2);
P = zeros(1,dim);  Q = P;
for v1 = 1:size(X,1)
    for v2 = v1+1:size(X,1)
        [d,idx] = sort(abs(X(v1,:) - X(v2,:)));     
        if Y(v1)==Y(v2)
            P(idx(1:round(dim*percDim))) = P(idx(1:round(dim*percDim))) + 1;
        else
            Q(idx(end - round(dim*percDim)+1:end)) = Q(idx(end - round(dim*percDim)+1:end)) + 1;
        end
    end
end
classes = unique(Y)';
for v = classes
    numSamples(v) = sum(Y==v);
end
Np = 0;     Nq = 0;
for v1 = classes
    Np = Np + (numSamples(v1)*(numSamples(v1)-1))/2;
    for v2 = v1+1:length(classes)
        Nq = Nq + numSamples(v1)*numSamples(v2);
    end
end
P = P/Np;
Q = Q/Nq;

r1 = abs(P-Q)./(P+Q);
[~,rankx] = sort(r1);