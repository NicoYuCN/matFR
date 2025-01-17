from FSF import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This is an example of how to use the Feature-Ranking Self-Growing Forest
# algorithm (FSF) for classification tasks.
# This software was created in 2022 by Ruben I. Carino-Escobar, Gustavo A. Alonso-Silverio,
# Antonio Alarcon-Paredes and Jessica Cantillo-Negrete.

# The criotherapy dataset, by Khozeimeh et al.(F. Khozeimeh, R. Alizadehsani, M. Roshanzamir, A. Khosravi, P. Layegh, and S. Nahavandi, 'An expert system for selecting wart treatment method,'ç
# Computers in Biology and Medicine, vol. 81, pp. 167-175, 2/1/ 2017. and F. Khozeimeh, F. Jabbari Azad, Y. Mahboubi Oskouei, M. Jafari, S. Tehranian, R. Alizadehsani, et al.,
# 'Intralesional immunotherapy compared to cryotherapy in the treatment of warts,' International Journal of Dermatology, 2017, DOI: 10.1111/ijd.13535) 
# is loaded. This data set has 6 predictors and 90 observations. The outputs are binary treatment results.

# Load Cryotherapy Classification dataset
from scipy.io import loadmat
data = loadmat('cryotherapy.mat')
# print(data.keys())
X = data['cryotherapy_tot']
y = data['cryotherapy_class']
y = np.reshape(y,-1)

# 80% of dataset is used for training, and 20% for testing
# Data is randomly partitioned into training and testing subsets 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8, shuffle=True, random_state=23)

mdl = FSF(verbose=False) #FSF Object instance

# The only INPUTS that the FSF needs is the training matrix X_train
# and the class vector y_train. OUTPUTS are the forest structure "forest" and
# the histogram of the most important features "feature_imp".
forest, feature_imp = mdl.fit(X_train, y_train)
# For performing a classification with the trained FSF model "forest", the predict method is used.
# INPUT parameters are the test matrix Xtest, the forest structure previously built,
# and the third argument is 1 or "Classification" for classification, and 2 or "Regression" for regression.
yhat = mdl.predict(X_test,forest,task='Classification')
# Classification accuracy with the FSF is computed. Output may vary every time this programm is executed.
score = mdl.classification_score(X_test,y_test)

print('\nFSF')
print(f'Accuracy: {score}')
print(f'Forest size: {len(mdl.forest_)}')

# An histogram with the most important features used for prediction
# according to the FSF criteria is plotted. This will change every time the
# program is executed. The feature_imp vector contains the relative frequencies
# of each predictor according to the FSF criteria. This vector will always
# sum 100, since importance is expresed in percentages.
plt.figure(figsize=(10,6))
plt.bar(range(len(feature_imp)),feature_imp)
plt.xticks(ticks=range(len(feature_imp)),labels=['Sex','Age','Time','Number of Warts','Type','Area'])
plt.title('Feature importance for classification of Cryotherapy outcomes')
plt.xlabel('Features')
plt.ylabel('Importance (%)')
plt.show()
