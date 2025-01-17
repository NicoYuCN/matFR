clear
close all
clc

%This is an example of how to use the Feature-Ranking 
%Self-Growing Forest algorithm (FSF) for classification and for regression.
%This program was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
%Antonio Alarcon-Paredes and Jessica Cantillo-Negrete. This code is
%licensed under the MIT license, and can be donwloaded from the following
%GitHub project: 

%1. Classification
%The criotherapy dataset by Khozeimeh et al.(F. Khozeimeh, R. Alizadehsani, M. Roshanzamir, A. Khosravi, P. Layegh, and S. Nahavandi, 'An expert system for selecting wart treatment method,' Computers in Biology and Medicine, vol. 81, pp. 167-175, 2/1/ 2017. and F. Khozeimeh, F. Jabbari Azad, Y. Mahboubi Oskouei, M. Jafari, S. Tehranian, R. Alizadehsani, et al., 'Intralesional immunotherapy compared to cryotherapy in the treatment of warts,' International Journal of Dermatology, 2017, DOI: 10.1111/ijd.13535) 
%is loaded. This data set has 6 predictors and 90 observations. The output
%are binary treatment results. 

load cryotherapy
class=cryotherapy_class;
X=cryotherapy_tot;


%80% of dataset is used for training, and 20% for testing
%Data is randomly partiotioned into training and testing subsets 
obs=length(X);
rperm=randperm(obs);
X=X(rperm,:);
class=class(rperm,:);
itrain=round(obs*0.8);
Xtrain=X(1:itrain,:);
Ytrain=class(1:itrain,:);
Xtest=X(itrain+1:end,:);
Yreal=class(itrain+1:end,:);

%The only parameters that the FSF needs is the training matrix Xtrain
%and the class vector Ytrain. Outputs are the forest structure "forest" and
%the histogram of the most important features "featimp".
[forest,featimp] = FSF(Xtrain,Ytrain);

%For performing a classification with the trained FSF model "forest", the
%FSF prediction function is used. Input parameters are the test matrix
%Xtest, the forest structure previously computed "forest",
%and the third argument is "1" for classification and "2" for regression.
%In this case we use classification.
YFSF=FSF_prediction(Xtest,forest,1);

%Classification accuracy with the FSF is computed. Output will have a
%variance every time this programm is executed.
result=((length(find(Yreal==YFSF))*100)/length(YFSF));
disp(' ')
disp(['The classification accuracy for predicting cryotherapy outcomes with the FSF algorithm was =',num2str(result)])
disp(' ')

%An histogram with the most important features used for prediction
%according to the FSF criteria is plotted. This will change every time the
%program is executed. The featimp vector contains the relative frequencies
%of each predictor according to the FSF criteria. This vector will always
%sum 100, since importance is expresed in percentages.
figure(1)
X = categorical({'Sex','Age','Time','Number of Warts','Type','Area'});
bar(X,featimp);
title('Feature importance for classification of Cryotherapy outcomes')
xlabel('Features')
ylabel('Importance (%)')


%2. Regression
%The Yatch Hydrodynamics Dataset by Dr. Roberto Lopez (I. Ortigosa, R. Lopez and J. Garcia. A neural networks approach to residuary resistance of sailing
%yachts prediction. In Proceedings of the International Conference on Marine Engineering MARINE 2007, 2007.
%%The output is the price of houses 
load yatchreg
class=yatchreg_class;
X=yatchreg_tot;

%80% of dataset is used for training, and 20% for testing
%Data is randomly partiotioned into training and testing subsets 
obs=length(X);
rperm=randperm(obs);
X=X(rperm,:);
class=class(rperm,:);
itrain=round(obs*0.8);
Xtrain=X(1:itrain,:);
Ytrain=class(1:itrain,:);
Xtest=X(itrain+1:end,:);
Yreal=class(itrain+1:end,:);

%The only parameters that the FSF needs is the training matrix Xtrain
%and the class vector Ytrain Outputs are the forest structure "forest" and
%the histogram of the most important features "featimp".
[forest,featimp] = FSF(Xtrain,Ytrain);
%For performing a classification with the trained FSF model "forest", the
%FSF prediction function is used. Input parameters are the test matrix
%Xtest, the forest structure previously obtained with the forest structure,
%and the third argument is "1" for classification and "2" for regression.
%We put "2" since in this case we have a regression task.
YFSF=FSF_prediction(Xtest,forest,2);

%Mean absolute Error is computed with the FSF. Output will have a
%variance every time this programm is executed.
result=(sum(abs(Yreal-YFSF)))/(length(Yreal));
disp(' ')
disp(['The Mean absolute error for predicting residuary resistance of sailing yachts =',num2str(result)])

%An histogram with the most important features used for prediction
%according to the FSF criteria is plotted. This will change every time the
%program is executed. The featimp vector contains the relative frequencies
%of each predictor according to the FSF criteria. This vector will always
%sum 100, since importance is expresed in percentages.
figure(2)
X = categorical({'Longitudinal position of the center of buoyancy','Prismatic coefficient','Length-displacement ratio','Beam-draught ratio','Length-beam ratio','Froude number'});
bar(X,featimp);
title('Feature importance for regression of Residuary resistance per unit weight of displacement')
xlabel('Features')
ylabel('Importance (%)')