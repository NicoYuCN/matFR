//Nguyen X. Vinh, Jeffrey Chan, Simone Romano and James Bailey, "Effective Global Approaches for Mutual Information based Feature Selection". 
//To appear in Proceeedings of the 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'14), August 24-27, New York City, 2014. 
// (C) 2014 Nguyen Xuan Vinh   
// Email: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
// Mex code for computing the JMI matrix: last row of data matrix is class variable

#include "mex.h" /* Always include this */
#include <math.h>
#include <iostream>

using namespace std;

int compare_feature_config(double *data,int nPa,int* Pa,int a,int posi, int posj);
double conditional_MI(double *data,int i, int j,int nPa,int *Pa, int n_state,int n_stateC);
void  Contingency(int Mem1,int Mem2,int n_state,int n_stateC);
double Mutu_Info(int **T, int n_state,int n_stateC);
void ClearT(int n_state,int n_stateC);			//clear the share contingency table
double conditional_MI_fast(double *data,int a, int b,int nPa, int* Pa, int n_state,int n_stateC);
double getij(double* A, int nrows,int i,int j){return A[i + nrows*j];}
void setij(double* A, int nrows,int i,int j,double val){A[i + nrows*j]=val;}


//global variables
int** T=NULL;
int*** newT=NULL;
int n_state=0;
int n_stateC=0;
int N=0;
int dim=0;
double* data=NULL;     


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

#define A_IN prhs[0]
#define B_OUT plhs[0]

double *B, *A;       
N = mxGetM(A_IN); /* Get the dimensions of A */
dim = mxGetN(A_IN);

B_OUT = mxCreateDoubleMatrix(dim-1, dim-1, mxREAL); /* Create the output matrix */

data = mxGetPr(A_IN);

B = mxGetPr(B_OUT); /* Get the pointer to the data of B */
//get number of state of the data
for(int i=0;i<N;i++){
	   for (int j=0;j<dim-1;j++){			
            if (n_state<data[i+N*j]) n_state=data[i+N*j];
        }
}

for(int i=0;i<N;i++){
    if (n_stateC<data[i+(dim-1)*N]) n_stateC=data[i+(dim-1)*N];
}

n_state++;  //Attention: buffer for state (C++ index starts from 0)
n_stateC++;
        
T=new int*[n_state];
for(int i=0;i<n_state;i++) T[i]=new int[n_stateC];

newT=new int**[n_state]; //for computing I(Xi;C|Xj)
for(int i=0;i<n_state;i++){
    newT[i]=new int* [n_state];
    for(int j=0;j<n_state;j++) newT[i][j]=new int [n_stateC];
}

cout<<"Computing CMI matrix v4 (fast) started...\n";

//compute JMI and fill in B
int* Pa=new int[1];

for (int i = 0; i < dim-1; i++){ /* Compute a matrix with normalized columns */
    double MI1=conditional_MI(data, i, dim-1,0, Pa, n_state, n_stateC);
    Pa[0]=i;
    //setij(B,dim-1,i,i,MI1);
    B[i+(dim-1)*i]=MI1;
    for(int j = 0; j < dim-1; j++) /* Compute a matrix with normalized columns */
    {
        if(j==i) continue;
        //double MI=conditional_MI(data, j, dim-1 ,1, Pa, n_state, n_stateC);
        double MI=conditional_MI_fast(data, j, dim-1 ,1, Pa, n_state, n_stateC);
        //printf(" MI=%f MI1= %f\n",MI,MI1);
        //setij(B,dim-1,i,j,MI+MI1);
        //setij(B,dim-1,j,i,MI+MI1);
        //setij(B,dim-1,i,j,MI);
        B[i+(dim-1)*j]=MI;
    }
    //mexPrintf("\n");
}


for(int i=0;i<n_state;i++) delete[] T[i];
delete T;

for(int i=0;i<n_state;i++) {
    for(int j=0;j<n_state;j++) delete[] newT[i][j];
    delete[] newT[i];
}
delete newT;


return;
}


// data[i][j] => data[i + nrows*j];
void Contingency(int a,int b,int n_state,int n_stateC){	
    for(int i=0;i<n_state;i++)
        for(int j=0;j<n_stateC;j++)
            T[i][j]=0;    
    //build table
	for(int i =0;i<N;i++){         
		 //T[data[i][a]][data[i][b]]++;  
        T[(int)data[i+N*a]][(int)data[i+N*b]]++;  
	}   
}

//compare a feature set configuration of node a at two position in the data: char type
int compare_feature_config(double *data,int nPa,int* Pa,int a,int posi, int posj){
	int	isSame=1;
	for (int i=0;i<nPa;i++){ //scan through the list of features        
		//if(data[posi][Pa[i]]!=data[posj][Pa[i]]){//check this feature value at posi & posj
        if(data[posi+N*Pa[i]]!=data[posj+N*Pa[i]]){//check this feature value at posi & posj
			return 0;
		}
	}
return isSame;
}


void ClearT(int n_state,int n_stateC){
	for(int i=0;i<n_state;i++){
		for(int j=0;j<n_stateC;j++){
			T[i][j]=0;
		}
	}
}

//conditional MI between node a-> node b given other feature Pa
double conditional_MI(double *data,int a, int b,int nPa, int* Pa, int n_state,int n_stateC){
double MI=0;

if (nPa==0){ //no feature
     Contingency(a,b,n_state, n_stateC);
	 return Mutu_Info(T, n_state, n_stateC);
}
else {	//with some features?
	int  * scanned=new int[N];

	for(int i=0;i<N;i++){scanned[i]=0;}

	for(int i=0;i<N;i++){ //scan all rows of data
		if(scanned[i]==0){  //a new  combination of Pa found		
			scanned[i]=1;
			double count=1;
			ClearT(n_state, n_stateC);
			T[(int)data[i+N*a]][(int)data[i+N*b]]++;

			for(int j=i+1;j<N;j++){
				if(scanned[j]==0 && data[i+N*Pa[0]]==data[j+N*Pa[0]]){ //   compare_feature_config(data,nPa,Pa,b,i,j)){
					scanned[j]=1;	 				
					T[(int)data[j+N*a]][(int)data[j+N*b]]++;
					count++;
				}
			}
			MI+=(count/N)*Mutu_Info(T,n_state, n_stateC);
		}
	}
	delete[] scanned;	
}

return MI;
}


//conditional MI between node a-> node b given other feature Pa
//fast version (required more memory)
double conditional_MI_fast(double *data,int a, int b,int nPa, int* Pa, int n_state,int n_stateC){
double MI=0;

if (nPa==0){ //no parents
     Contingency(a,b,n_state, n_stateC);
	 return Mutu_Info(T, n_state, n_stateC);
}
else {	//with some parents
    for(int i=0;i<n_state;i++) for(int j=1;j<n_state;j++) for(int k=1;k<n_stateC;k++) newT[i][j][k]=0;
    
    double* count=new double[n_state];
    for (int i=0;i<n_state;i++) count[i]=0;
    
	for(int i=0;i<N;i++){ //scan all rows of data
        int cl=(int) data[i+N*Pa[0]];
        newT[cl][(int)data[i+N*a]][(int)data[i+N*b]]++;
        count[cl]++;        
    }
    
    for(int i=1;i<n_state;i++) MI+=(count[i]/N)*Mutu_Info(newT[i],n_state, n_stateC);
    
    delete count;
}

return MI;
}

double Mutu_Info(int **T, int n_state,int n_stateC){  //get the mutual information from a contingency table
	//n_state: #rows n_stateC:#cols
    double MI=0;
	int *a = new int[n_state];
	int *b = new int[n_stateC];
	int N=0;

	for(int i=1;i<n_state;i++){ //row sum
		a[i]=0;
		for(int j=1;j<n_stateC;j++)
		{a[i]+=T[i][j];}
	}

	for(int i=1;i<n_stateC;i++){ //col sum
		b[i]=0;
		for(int j=1;j<n_state;j++)
		{b[i]+=T[j][i];}
	}

	for(int i=1;i<n_state;i++) {N+=a[i];}
    
	for(int i=1;i<n_state;i++){
		for(int j=1;j<n_stateC;j++){
			if(T[i][j]>0){
				MI+= T[i][j]*log(double(T[i][j])*N/a[i]/b[j])/log(double(2));
			}
		}
	}
	delete []a;
	delete []b;

	if(N>0) 	return MI/N;
	else return 0;
}