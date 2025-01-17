import numpy as np
from collections.abc import Sized
from random import random
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import mode as st_mode
from warnings import filterwarnings
filterwarnings("ignore")

# This class corresponds to the Feature-Ranking Self-Growing Forest
# algorithm (FSF) for classification and regression tasks.
# This software was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
# Antonio Alarcón-Paredes and Jessica Cantillo-Negrete.

class FSF():
    def __init__(self, verbose=False):
        self.forest_ = None
        self.task_ = None
        self.verbose = verbose
        pass


    def fit(self, X, y):
        stop_criteria=0
        try:
            obs, k = X.shape
        except:
            print('[HALT!] Algorithm not executed: input data needs to have more than 1 observation (row) and more than one predictor feature (column)')
            exit()
        ntree=0
        forest=[]
        contsize=0
        past_forest_size=0
        avgfreq_past=0
        cont=1

        while stop_criteria==0:
            if obs<2 or k<2:
                print('Algorithm not executed: input data needs to have more than 1 observation (row) and more than one predictor feature (column)')
                stop_criteria = 1
                feature_imp = []
                exit()

            prob = np.random.randint(0,100,k)
            indx = np.argwhere(prob>=50).T[0]
            if len(indx)<3:
                indx = [i for i in range(0,k)]
                X_temp = X
            else:
                X_temp = X[:,indx]
            
            forest.append(self.__FSF_tree(X_temp, y, indx))
            ntree+=1
            contsize+=1
            if contsize >= (past_forest_size/10):
                contsize=0
                past_forest_size = ntree
                sal=[]
                for i in range(len(forest)):
                    saltemp = forest[i][0]
                    sal.append(saltemp)
                    nleft = len(forest[i][2]) if isinstance(forest[i][2], Sized) else 1
                    nright = len(forest[i][3]) if isinstance(forest[i][3], Sized) else 1
                    if nleft>1 and nright==1:
                        sal.append(forest[i][2][0])
                    if nright>1 and nleft==1:
                        sal.append(forest[i][3][0])
                    if nright>1 and nleft>1:
                        sal.append(forest[i][2][0])
                        sal.append(forest[i][3][0])
                bins = [i for i in range(k+1)]
                histcounts,_ = np.histogram(sal, bins=bins)
                avgfreq = np.array(histcounts)*100 / np.sum(histcounts)
                maxdiff = np.max(abs(avgfreq - avgfreq_past))
                avgfreq_past = avgfreq
                if self.verbose:
                    print('The forest size is currently:',str(len(forest)),'trees.')
                    print('Maximum difference between predictor variables frequencies between the current and past forest is:',str(maxdiff)+';', 'target is less than 1')
            if cont>10:
                if maxdiff<1:
                    stop_criteria = 1
                    feature_imp = avgfreq
            cont+=1
            self.forest_ = forest
        return forest, feature_imp
    

    def predict(self, X, forest, task=1):
        # On argument "task", 1 or 'Classification' is for classification, 2 or 'Regression' is for regression
        self.task_ = task
        n,_ = X.shape
        yhat = np.zeros(n)
        if task==1 or task=='Classification':
            for j in range(n):
                ntree = len(forest)
                yi = np.zeros(ntree)
                for i in range(ntree):
                    onetree = forest[i]
                    indvar = onetree[0]
                    thresh = onetree[1]
                    ending = 0
                    Xi = X[j,:]
                    while ending==0:
                        if Xi[indvar] > thresh:
                            len_tree = len(onetree[3]) if isinstance(onetree[3], Sized) else 1
                            if len_tree==1:
                                yi[i] = onetree[3]
                                ending = 1
                            else:
                                onetree = onetree[3]
                                indvar = onetree[0]
                                thresh = onetree[1]
                        else:
                            len_tree = len(onetree[2]) if isinstance(onetree[2], Sized) else 1
                            if len_tree==1:
                                yi[i] = onetree[2]
                                ending = 1
                            else:
                                onetree = onetree[2]
                                indvar = onetree[0]
                                thresh = onetree[1]
                yhat[j] = st_mode(yi)[0][0]                
                yhat[j] = int(np.round(yhat[j],0))

        if task==2 or task=='Regression':
            for j in range(n):
                ntree = len(forest)
                yi = np.zeros(ntree)
                for i in range(ntree):
                    onetree = forest[i]
                    indvar = onetree[0]
                    thresh = onetree[1]
                    ending = 0
                    Xi = X[j,:]
                    while ending==0:
                        if Xi[indvar] > thresh:
                            len_tree = len(onetree[3]) if isinstance(onetree[3], Sized) else 1
                            if len_tree==1:
                                yi[i] = onetree[3]
                                ending = 1
                            else:
                                onetree = onetree[3]
                                indvar = onetree[0]
                                thresh = onetree[1]
                        else:
                            len_tree = len(onetree[2]) if isinstance(onetree[2], Sized) else 1
                            if len_tree==1:
                                yi[i] = onetree[2]
                                ending = 1
                            else:
                                onetree = onetree[2]
                                indvar = onetree[0]
                                thresh = onetree[1]
                _,p_value = lilliefors(yi)
                if p_value<0.05:
                    yhat[j] = np.median(yi)
                else:
                    yhat[j] = np.average(yi)
        return yhat


    def classification_score(self, X, y):
        if self.forest_ is None:
            print('You must fit the model first')
            exit()
        else:
            yhat = self.predict(X, self.forest_, task=1)
            obs = len(y)
            score = int(obs - np.sum(abs(yhat-y))) / obs
            return score
    

    def regression_score(self, X, y):
        if self.forest_ is None:
            print('You must fit the model first')
            exit()
        else:
            yhat = self.predict(X, self.forest_, task=2)
            score = mean_absolute_error(yhat, y)
            return score


    def __FSF_rand_split(self, X, feat):
        # Take the feature 'feat' individually
        Xfeat = X[:, feat]
        maxi = max(Xfeat)
        mini = min(Xfeat)
        # Search for the threshold cut
        valsplit = random() * (maxi - mini) + mini
        SR = np.argwhere(Xfeat>valsplit).T[0]
        SL = np.argwhere(Xfeat<=valsplit).T[0]
        return valsplit, SR, SL
    

    def __FSF_tree(self, X, y, indx):
        _,feat = X.shape
        cut=[]
        SR=[]
        SL=[]
        score=[]

        if len(np.unique(y))==1 or feat<2:
            tree = np.mean(y)
        else:
            for i in range(feat):
                cuti,SRi,SLi = self.__FSF_rand_split(X,i)
                cut.append(cuti)
                SR.append(SRi)
                SL.append(SLi)
                scori = abs(np.mean(y[SRi]) - np.mean(y[SLi]))
                
                if np.isnan(scori):
                    score.append(-1e6)
                else:
                    score.append(scori)
            iselec = list(np.where(score==np.max(score))[0])

            l = len(iselec)
            if l>1:
                l = np.random.randint(0,l)
                iselec = [iselec[l]]

            YR = y[SR[iselec[0]]]
            YL = y[SL[iselec[0]]]
            XR = X[SR[iselec[0]]]
            XL = X[SL[iselec[0]]]

            tree = list(np.zeros(4))
            if np.isnan(cut[iselec[0]]) or not SR[iselec[0]].size or not SL[iselec[0]].size:
                tree[0] = indx[iselec[0]]
                tree[1] = cut[iselec[0]]

                if not SL[iselec[0]].size:
                    tree[2] = y[np.random.randint(0,len(y))]
                else:
                    tree[2] = np.mean(y[SL[iselec[0]]])
                
                if not SR[iselec[0]].size:
                    tree[3] = y[np.random.randint(0,len(y))]
                else:
                    tree[3] = np.mean(y[SR[iselec[0]]])
            else:
                treeR = self.__FSF_tree(XR, YR, indx)
                treeL = self.__FSF_tree(XL, YL, indx)

                tree[0] = indx[iselec[0]]
                tree[1] = cut[iselec[0]]
                tree[2] = treeL
                tree[3] = treeR
        return tree
    

    def holdout(self,X,y,train_size=0.5, random_state=42):
        np.random.seed(random_state)
        obs,_ = X.shape
        randperm = np.random.permutation(obs)
        X = X[randperm,:]
        y = y[randperm]
        itrain = round(obs*train_size)
        X_train = X[:itrain,:]
        X_test = X[itrain:,:]
        y_train = y[:itrain]
        y_test = y[itrain:]
        
        return X_train, X_test, y_train, y_test

    
if __name__ == '__main__':
    pass
