%This is function is used to apply a classification or regression model
%comprised by an ensemble of decision trees computed  the Feature-Ranking 
%Self-Growing Forest algorithm (FSF).
%This function was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
%Antonio Alarcon-Paredes and Jessica Cantillo-Negrete. This code is
%licensed under the MIT license, and can be downloaded from the following link
%https://github.com/RubenICarinoEscobar/Feature-Ranking-Self-Growing-Forest
%Input arguments are X a training matrix of data that we need to classify
%or regress (lines = features, columns = observations). A "forest"
%structure that is a cell array with each cell having an individual tree,
%this variable is an output of the "FSF" function. "Type" can be either 1
%or 2, and tells the function which tasks is applied, 1=classification,
%2=regression. "Outtot" is a vector with the classes of continous values of
%the outputs.
function [outtot] = FSF_prediction(X,forest,type)

[nobs,~]=size(X);
outtot=zeros([nobs,1]);
%Classification
if type==1
    %For every observation compute an output with each tree
    for j=1:nobs
        ntree=length(forest);
        outf=zeros([ntree,1]);
        %For each tree within the ensemble stores in "forest" compute an
        %output
            for i=1:ntree
                    onetree=forest{i};
                    indvar=onetree{1};
                    thresh=onetree{2};
                    ending=0;
                    Xi=X(j,:);
                    %Continue following the tree's branches until a leaf
                    %node is reached
                    while ending==0
                        if Xi(indvar)>thresh
                            %If the right node is chosen and it is a leaf
                            %node
                            if length(onetree{4})==1
                                a=onetree{4};
                                outf(i)=a{1};
                                ending=1;
                            %If the right node is chosen and it is not a leaf
                            %node
                            else
                                onetree=onetree{4};
                                indvar=onetree{1};
                                thresh=onetree{2};
                            end  
                        else
                            %If the left node is chosen and it is a leaf
                            %node
                            if length(onetree{3})==1
                                a=onetree{3};
                                outf(i)=a{1};
                                ending=1;
                            %If the left node is chosen and it is not a leaf
                            %node
                            else
                                onetree=onetree{3};
                                indvar=onetree{1};
                                thresh=onetree{2};
                            end

                        end

                    end
            end
        %The majority voting of the trees is chosen as the final decision
        outtot(j)=mode(outf);
        outtot(j)=round(outtot(j));
    end
    
end


%Regression
if type==2
    %For every observation compute an output with each tree 
    for j=1:nobs
        ntree=length(forest);
        outf=zeros([ntree,1]);
        %For each tree within the ensemble stores in "forest" compute an
        %output
            for i=1:ntree
                    onetree=forest{i};
                    indvar=onetree{1};
                    thresh=onetree{2};
                    ending=0;
                    Xi=X(j,:);
                    %Continue following the tree's branches until a leaf
                    %node is reached
                    while ending==0
                        if Xi(indvar)>thresh
                            %If the right node is chosen and it is a leaf
                            %node
                            if length(onetree{4})==1
                                a=onetree{4};
                                outf(i)=a{1};
                                ending=1;
                            else
                            %If the right node is chosen and it is not a leaf
                            %node
                                onetree=onetree{4};
                                indvar=onetree{1};
                                thresh=onetree{2};
                            end  
                        else
                            %If the left node is chosen and it is a leaf
                            %node
                            if length(onetree{3})==1
                                a=onetree{3};
                                outf(i)=a{1};
                                ending=1;
                            else
                            %If the left node is chosen and it is not a leaf
                            %node
                                onetree=onetree{3};
                                indvar=onetree{1};
                                thresh=onetree{2};
                            end

                        end

                    end
            end
        warning('off','all')
        [h,~]=lillietest(outf);
        warning('on','all')
        %The average of the trees' output is chosen as the final decision
        if h==1    
        outtot(j)=median(outf);
        else
        outtot(j)=mean(outf);    
        end
    end
    
end
