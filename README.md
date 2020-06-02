# matFR

matFR: A matlab toolbox for feature ranking

Background: 

  To collect thousands of features becomes feasible in medical imaging and biology, and thus, 
how to figure out these informative and discriminative ones becomes urgently important. 
  The purpose of this project is to collect feature ranking (FR) methods, to provide a platform 
easy-to-use for potential researchers, and to call for designers online-sharing their previously 
or newly developed algorithms.

Methods: 

  The matFR toolbox has already integrated 42 methods. Among them, 12 methods are from FSLib [1], 
9 methods from mutual information (MI) based feature selection repository [2], 7 methods are used 
in MATLAB (“rankfeatures”, “relieff” and “lasso”), and others are accessible online. 
  FR methods can be grouped into supervised and unsupervised methods. In this toolbox, there are 29 
supervised and 13 unsupervised methods. 
  FR methods can also be categorized from theoretical perspective. The matFR contains 12 MI based 
methods, 8 statistical analysis based methods, 8 structure learning based methods and others.

An example:

  Given an input matrix X (m instances and n features per instance) and its corresponding labels Y 
(Y ∈ {0, 1}), the procedure of using one (f) of the FR methods (F) in the toolbox can be described 
as r = F(X, f, Y ), where r stands for the output rank indexes of features in a descending order 
with regard to the relative importance of features. If an unsupervised method is selected, the class 
labels Y can be omitted.


Implementation:

The toolbox is mainly implemented with MATLAB, while the MI based methods require a C++ compiler to 
compile two cpp files. One file aims for computing the pairwise MI matrix between feature-feature and 
feature-class, and the other is for the joint MI matrix. 

The matFR toolbox has been tested on 64-bit Windows 7/8/10 systems, MATLAB R2018a/R2019a and Microsoft 
Visual C++ 2012/2015/2017.


Future work

The future work arises from three aspects. First, to integrate available FR methods into the toolbox and 
to follow up newly developed methods. The most promising way is contributions from algorithm developers 
to this project through online collaboration. Second, to complete the details of FR methods. For instance, 
some advanced discretization algorithms (CAIR [3] and CAIM [4]) could be adopted to MI based methods for 
diverse options. Last but not the least, to accelerate the distribution of these FR algorithms, the toolbox 
could be implemented in Python and R.

Reference:

[1] Roffo, G. (2016) Feature selection library (MATLAB toolbox), arXiv preprint, arXiv, 1607.01327.

[2] Nguyen, X.V., Chan, J., Romano, S. and Bailey, J. (2014) Effective global approaches for mutual 
    information based feature selection, 20th ACM SIGKDD international conference on knowledge discovery 
    and data mining, 512-521.

[3] Ching, J.Y., Wong, A.K.C., Chan, K.C.C. (1995) Class-dependent discretization for inductive learning 
from continuous and mixed-mode data, IEEE Transactions on Pattern Analysis and Machine Intelligence, 17(7), 
641-651.

[4] Kurgan, L.A., Cios, K.J. (2004) CAIM discretization algorithm, IEEE transactions on Knowledge and Data 
Engineering, 16(2), 145-153.
    
