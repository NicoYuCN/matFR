from FSF import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# This is an example of how to use the Feature-Ranking Self-Growing Forest
# algorithm (FSF) for classification tasks.
# This software was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
# Antonio Alarcón-Paredes and Jessica Cantillo-Negrete.

# The Yatch Hydrodynamics Dataset by Dr. Roberto Lopez (I. Ortigosa, R. Lopez and J. Garcia.
# A neural networks approach to residuary resistance of sailing yachts prediction.
# In Proceedings of the International Conference on Marine Engineering MARINE 2007, 2007.
# The output is the residuary resistance per unit weight of displacement.

# Load YatchReg Regression dataset
from scipy.io import loadmat
data = loadmat('yatchreg.mat')
# print(data.keys())
X = data['yatchreg_tot']
y = data['yatchreg_class']
y = np.reshape(y,-1)

# 80% of dataset is used for training, and 20% for testing.
# Data is randomly partiotioned into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8, shuffle=True, random_state=23)

mdl = FSF(verbose=False) #FSF Object instance

# The only INPUTS that the FSF needs is the training matrix X_train
# and the class vector y_train. OUTPUTS are the forest structure "forest" and
# the histogram of the most important features "feature_imp".
forest, feature_imp = mdl.fit(X_train, y_train)
# For performing a classification with the trained FSF model "forest", the predict method is used.
# INPUT parameters are the test matrix Xtest, the forest structure previously built,
# and the third argument is 1 or "Classification" for classification, and 2 or "Regression" for regression.
yhat = mdl.predict(X_test,forest,task=2)
# Mean absolute Error is computed with the FSF. Output may vary every time this programm is executed.
score = mdl.regression_score(X_test, y_test)

print('\nFSF')
print(f'Mean absolute error: {score}')
print(f'Forest size: {len(mdl.forest_)}')

# An histogram with the most important features used for prediction
# according to the FSF criteria is plotted. This will change every time the
# program is executed. The feature_imp vector contains the relative frequencies
# of each predictor according to the FSF criteria. This vector will always
# sum 100, since importance is expresed in percentages.
plt.figure(figsize=(10,6))
plt.bar(range(len(feature_imp)),feature_imp)
plt.xticks(ticks=range(len(feature_imp)),labels=['Long pos of center\nof buoyancy','Prismatic\ncoefficient','Len-displacement\nratio','Beam-draught\nratio','Length-beam\nratio','Froude number'])
plt.title('Feature importance for regression of \nResiduary resistance per unit weight of displacement')
plt.xlabel('Features')
plt.ylabel('Importance (%)')
plt.show()
