%This is an auxiliary function used to compute the Feature-Ranking 
%Self-Growing Forest algorithm (FSF).
%This function was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
%Antonio Alarcon-Paredes and Jessica Cantillo-Negrete. This code is
%licensed under the MIT license, and can be downloaded from the following link
%https://github.com/RubenICarinoEscobar/Feature-Ranking-Self-Growing-Forest
%Input arguments are X a training matrix of training data
% (lines = features, columns = observations). Y is a vector with the training
% outputs of classess or continous values.
%feat is the index of the feature, drawn from the original feature space
%Outputs are "valsplit" a scalar with the threshold for splitting nodes,
%splitright the indexes of the observations that for the right node (lines=
%for each feature). Splileft the same as splitright but for the left node.


function [valsplit,splitright,splitleft] = FSF_rand_split(X,Y,feat)

Xfeat=X(:,feat);
maxXfeat=max(Xfeat);
minXfeat=min(Xfeat);
%The range of values or classes is computed
a=[minXfeat, maxXfeat];
out=0;
contout=0;
%For 100 iterations try to find a threshold "cut" that can partition the
%observations on the given feature into at least 2 subsets
while out==0
    
    cut=((a(1,1)-a(1,2)))*rand + a(1,2);

    splitleft=zeros([length(Y),1]);
    splitright=zeros([length(Y),1]);
    right=1;
    left=1;
    for j=1:length(Y)
        if X(j,feat) > cut
            splitright(right)=j;
            right=right+1;
        else
            splitleft(left)=j;
            left=left+1;
        end
    end
     
    %If a cut is found then the function stops
    if sum(splitright)>0 && sum(splitleft)>0
    ir=find(splitright==0);
    il=find(splitleft==0);
    splitright(ir)=[];
    splitleft(il)=[];
    right=cell(1);
    left=cell(1);
    right{1}=splitright;
    left{1}=splitleft;
    splitright=right;
    splitleft=left;
    valsplit=cut;
    out=1;
    end
    
    contout=contout+1;
    %If after 100 iterations a cut is not found then the function stops and
    %returns an empty arreay.
    if contout>=100
    ir=find(splitright==0);
    il=find(splitleft==0);
    splitright(ir)=[];
    splitleft(il)=[];
    right=cell(1);
    left=cell(1);
    right{1}=splitright;
    left{1}=splitleft;
    splitright=right;
    splitleft=left;
    valsplit=cut;
    out=1;
    end

end

end
