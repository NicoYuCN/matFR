%This is an auxiliary function used to compute the Feature-Ranking 
%Self-Growing Forest algorithm (FSF), specifically each decision tree within the ensemble.
%This function was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
%Antonio Alarcon-Paredes and Jessica Cantillo-Negrete. This code is
%licensed under the MIT license, and can be downloaded from the following link
%https://github.com/RubenICarinoEscobar/Feature-Ranking-Self-Growing-Forest
%Input arguments are X a training matrix of training data
% (lines = features, columns = observations). Y is a vector with the training
% outputs of classess or continous values.
%indx is the index of the features, drawn from the original feature space
%Output is "tree" a cell array that contains the tree structure. The first
%cell is a value, the index of the variable, the second value is the
%threshold's value, the left node, a value if it is a leaf node and another
%cell if it is a non-leaf node. The same for the 4th cell but for the right
%node.

function [tree] = FSF_tree(X,Y,indx)
[~,k]=size(X);
%If the number of observations is 1 or if the features are less than 2 then
%this is a leaf node and the output of the observations is averaged
  if  length(unique(Y))==1 || k<2
        tree={mean(Y)} ;  
    else

    splitright=cell([k,1]);
    splitleft=cell([k,1]);
    cut=zeros([k,1]);
    %A threshold "cut" is computed for each feature
    for i=1:k
        
    [cut(i),splitright(i),splitleft(i)]=FSF_rand_split(X,Y,i);
        
    end
    score=zeros(length(k));
    
    %Each feature is evaluated using a fitness criterion that can be used
    %for both classification or regression
    for i=1:k

    Yleft=splitleft{i}; 
    Yright=splitright{i};
    
    %Fitness criterion, the higher it is the better separation of
    %observations with the feature with index "i", will be achieved
    score(i)= abs (mean(Y(Yleft)) - mean(Y(Yright)));
    
    %If there was not a threshold that could separate data into 2 subsets,
    %then it was a bad performing feature and the lowest fitness value is
    %assigned to it
    if isnan(cut(i))==1
    score(i)=-1000000;     
    end

    end
   
    %The best feature for splitting the data more efficiently is selected
    %for constructing the node
    iselec=find(score==max(score));
    %If none of the thresholds could split data into 2 subsets, or if  two
    %variables were better at splitting data then one of them is randomly
    %chosen
    if isempty(iselec)==1
    iselec=randi(k);
    end
    if length(iselec)>1
    iselec=iselec(randi(length(iselec)));
    end
    YL=Y(splitleft{iselec});
    YR=Y(splitright{iselec});
    XL=X(splitleft{iselec},:);
    XR=X(splitright{iselec},:);

    %If some of the subsets are empty then this is a leaf node and compute
    %the average value of the node that it is not empty
     
 if  isnan(cut(iselec))== 1 || isempty(splitleft{iselec})==1 || isempty(splitright{iselec})==1
     tree{1}=indx(iselec);
     tree{2}=cut(iselec);
     
     if isempty(splitleft{iselec})==1
     tree{3}={Y(randi(length(Y)))};  
     else
     tree{3}={mean(Y(splitleft{iselec}))};
     end
     
     if isempty(splitright{iselec})==1
     tree{4}={Y(randi(length(Y)))};
     else
     tree{4}={mean(Y(splitright{iselec}))};
     end
 else  
     %This is not a leaf node, then call the function to create a new tree
     %and save the structure of the current node
     [treeL] = FSF_tree(XL,YL,indx);
     [treeR] = FSF_tree(XR,YR,indx);
     tree{1}=indx(iselec);
     tree{2}=cut(iselec);
     tree{3}=treeL;
     tree{4}=treeR;
 end
    
 end
 
end

