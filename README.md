
# matFR: A matlab toolbox for feature ranking




#### Background

  To collect thousands of features becomes feasible in medical imaging and biology, and thus, 
how to figure out these informative and discriminative ones becomes urgently important. 
  The purpose of this project is to collect feature ranking (FR) methods, to provide a platform 
easy-to-use for potential researchers, and to call for designers online-sharing their previously 
or newly developed algorithms.



#### Methods 

  The matFR toolbox has already integrated 42 methods. Among them, 12 methods are from FSLib [1], 
9 methods from mutual information (MI) based feature selection repository [2], 7 methods are used 
in MATLAB (“rankfeatures”, “relieff” and “lasso”), and others are accessible online. 
  FR methods can be grouped into supervised and unsupervised methods. In this toolbox, there are 29 
supervised and 13 unsupervised methods. 
  FR methods can also be categorized from theoretical perspective. The matFR contains 12 MI based 
methods, 8 statistical analysis based methods, 8 structure learning based methods and others.



#### An example

  Given an input matrix X (m instances and n features per instance) and its corresponding labels Y 
(Y ∈ {0, 1}), the procedure of using one (f) of the FR methods (F) in the toolbox can be described 
as r = F(X, f, Y ), where r stands for the output rank indexes of features in a descending order 
with regard to the relative importance of features. If an unsupervised method is selected, the class 
labels Y can be omitted.

  As shown in Figure under the folder "how2use", a user can activate any FR methods through an interface 
function ‘matFR_interface.m’ which determines the belonging of a method f. If f is a MI based FR method, 
the function ‘matFR_mi.m’ is triggered, else the other function ‘matFR_fn.m’ is activated.

  In details, an example is demonstrated. Given the BCDR-F03 data set (bcdr_data_whole.mat), it contains
406 lesions and 736 mammographic images [3]. To each annotated image, 17 features are 
selected to quantify the lesions from mass intensity, contour shape and lesion texture. Among the mass 
lesions, 230 are histologically verified benign (y = 0) and 176 are malignant (y = 1). Thus, to the input 
matrix X, m = 736 and n = 17. After the data is prepared, one line code to activate the Laplacian Score 
algorithm [4], an unsupervised method, is shown as below,

#####  r = matFR_interface ( X, ‘h2_fir_laplacian_score ’ );
           
and one line code to activate a joint MI based FR algorithm [5], a supervised method, 
is shown as below,

####  r = matFR_interface ( X, ‘b9_mi_joint ’ , Y).
            
Note that the short names of all FR methods are listed in the ‘demo.m’ file. As for further details, 
please refer to the publications and algorithm implementations.



#### Implementation

The toolbox is mainly implemented with MATLAB, while the MI based methods require a C++ compiler to 
compile two cpp files. One file aims for computing the pairwise MI matrix between feature-feature and 
feature-class, and the other is for the joint MI matrix. 

The matFR toolbox has been tested on 64-bit Windows 7/8/10 systems, MATLAB R2018a/R2019a and Microsoft 
Visual C++ 2012/2015/2017.



#### Future work

The future work arises from three aspects. First, to integrate available FR methods into the toolbox and 
to follow up newly developed methods. The most promising way is contributions from algorithm developers 
to this project through online collaboration. Second, to complete the details of FR methods. For instance, 
some advanced discretization algorithms (CAIR [6] and CAIM [7]) could be adopted to MI based methods for 
diverse options. Last but not the least, to accelerate the distribution of these FR algorithms, the toolbox 
could be implemented in Python and R.

#### Reference:

[1] Roffo, G. (2016) Feature selection library (MATLAB toolbox), arXiv preprint, arXiv, 1607.01327.

[2] Nguyen, X.V., Chan, J., Romano, S. and Bailey, J. (2014) Effective global approaches for mutual 
    information based feature selection, 20th ACM SIGKDD international conference on knowledge discovery 
    and data mining, 512-521.

[3] Arevalo, J., Gonzalez, F.A., Ramos-Pollan, R., Oliveira, J.L., Lopez, M.A.G. (2016) Representation 
learning for mammography mass lesion classification with convolutional neural networks, Computer methods 
and programs in biomedicine, 127, 248-257.

[4] He, X., Cai, D., Niyogi, P. (2006) Laplacian score for feature selection, Advances in neural information 
processing systems, 507-514.

[5] Nguyen, X.V., Chan, J., Romano, S. and Bailey, J. (2014) Effective global approaches for mutual information 
based feature selection, 20th ACM SIGKDD international conference on Knowledge discovery and data mining, 512-521.

[6] Ching, J.Y., Wong, A.K.C., Chan, K.C.C. (1995) Class-dependent discretization for inductive learning 
from continuous and mixed-mode data, IEEE Transactions on Pattern Analysis and Machine Intelligence, 17(7), 
641-651.

[7] Kurgan, L.A., Cios, K.J. (2004) CAIM discretization algorithm, IEEE transactions on Knowledge and Data 
Engineering, 16(2), 145-153.
    


#### if the matFR toolbox is useful, please refer to 

####   Zhang, Z., Liang, X., Yu, S. and Xie, Y. (2020) matFR: a matlab toolbox for feature ranking.
  
