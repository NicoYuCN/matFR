%This function is used to compute the Feature-Ranking 
%Self-Growing Forest algorithm (FSF).
%This function was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
%Antonio Alarcon-Paredes and Jessica Cantillo-Negrete. This code is
%licensed under the MIT license, and can be downloaded from the following link
%https://github.com/RubenICarinoEscobar/Feature-Ranking-Self-Growing-Forest
%Input arguments are X a training matrix of training data for the model
% (lines = features, columns = observations). Y is a vector with the training
% outputs of classess or continous values.
%Outputs are "forest_temp" a structure that is a cell array with each cell 
%having an individual tree, and "feature_imp", a vector with the importance
%of each feature expressed in percentage.

function [forest_temp,feature_imp] = FSF(X,Y)

cont=1;
ntree=1;
forest_temp=[];
stop_crit=0;
avgfreq_past=0;
[obs,k]=size(X);
nforest=1;
contsize=0;
pastforestsize=0;

%While the stop criteria is not reached, FSF will keep adding trees to the
%ensemble
while stop_crit==0
    
    %If there are less than 2 features or 2 observations then the function
    %does not execute and an error is displayed
    if obs<2 || k<2
    disp('Algorithm not executed, predictor input Matrix X needs to have more than 1 row (observation) and more than 1 column (predictor feature)');
    stop_crit=1; 
    feature_imp=[];
    return
    end
    
    %The forest cell array is created
    forest=cell([nforest,1]);
    prob=randi([1 100],[1, k]);
    X_temp1=[];
    indx1=[];
    %The probability of choosing a feature for computing a tree is 50%
    for i=1:k
        if prob(i)>=50
            X_temp1=[X_temp1, X(:,i)];
            indx1=[indx1, i];
        end
    end
    %If less than 3 features are chosen then all features are chosen to
    %avoid a very small tree
    if length(indx1)<3
        X_temp1=X;
        indx1=1:k;
    end
    forest_temp{ntree}=FSF_tree(X_temp1,Y,indx1);
    ntree=ntree+1;
    contsize=contsize+1;
    %Trees are aggregated to the ensemble until the new trees that were
    %added before reviewing the stop criteria are 10% of the past forest
    %size
    if contsize>=pastforestsize/10
    contsize=0;
    pastforestsize=ntree; 
        sal=[];
        %The frequency of features used in the first 3 nodes of each tree within the
        %ensemble is computed
        for i=1:length(forest_temp)
            saltemp=[forest_temp{i}{1}];
            sal=[sal, saltemp];
                     nleft=length(forest_temp{i}{3}); 
                     nright=length(forest_temp{i}{4});
                    if nleft>1 && nright==1
                    [ sal ] = [sal, forest_temp{i}{3}{1}];
                    end
                    if nright>1 && nleft==1
                    [ sal ] = [sal, forest_temp{i}{4}{1}];
                    end     
                    if nright>1 && nleft>1
                    sal = [sal,  forest_temp{i}{3}{1}, forest_temp{i}{4}{1}];    
                    end
        end
        sal=histcounts(sal,k);
        avgfreq=zeros(1,k);
        %The relative frequency of each feature in the first 3 nodes is expressed in percentage
        for i=1:k
        avgfreq(1,i)=(sal(i)*100)/sum(sal);
        end
        %The maximum difference of relative frequency for a feature is
        %computed
        maxdiff=max(abs(avgfreq-avgfreq_past));
        avgfreq_past=avgfreq;
        
        disp(['The forest size is currently  = ', num2str(length(forest_temp)),' Trees']);
        disp(['Maximum difference between predictor variables´ frequencies between the current and past forest is = ', num2str(maxdiff), ' target is less than 1']);
        
    end
    
       %If at least 10 iterantions passed, then the stopcriteria is
       %assessed
    if cont>10
        %If the maximum difference for any given feature is greater than 1
        %then the ensemble structural diversity remained unchanged even if
        %10% more trees were added, and the cycle stops.
        if maxdiff<1
            stop_crit=1;
            feature_imp=avgfreq;

        end    
    end
    
    cont=cont+1;
end

