#include <Rcpp.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include "linear.h"
using namespace Rcpp;

#define Malloc(type,n) (type *)Calloc(n,type)

void print_null(const char *s) {}

void setup_params(int *type, double *cost, double *epsilon, double* svr_eps, int *nrWi, double* C_vec, double *Wi, int *WiLabels, int *cross, int *verbose);
void setup_problem(double *X, double *Y, int *nbSamples, int *nbDim, int *sparse, int *rowindex, int *colindex, double *bi, int *verbose);
double do_cross_validation();

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int nr_fold;

/**
 * Function: trainLinear
 *
 * Author: Thibault Helleputte
 *
 */
void trainLinear(double *W_ret, int* labels_ret, double *X, double *Y, int *nbSamples, int *nbDim, double *C_vec, int *sparse, int *rowindex, int *colindex, 
                 double *bi, int *type, double *cost, double *epsilon, double* svr_eps, int *nrWi, double *Wi, int *WiLabels, int *cross, int *verbose)
{
	const char *error_msg;
	
	setup_params(type, cost, epsilon, svr_eps, nrWi, C_vec, Wi, WiLabels, cross, verbose);
	setup_problem(X, Y, nbSamples, nbDim, sparse, rowindex, colindex, bi, verbose);

	if(*verbose)
		Rprintf("SETUP CHECK\n");
	
	error_msg = check_parameter(&prob,&param);
	
	if(error_msg){
		Rprintf("ERROR: %s\n",error_msg);
		return;
	}
	
	if(flag_cross_validation)
	{
		if(*verbose)
			Rprintf("CROSS VAL\n");
		
		W_ret[0] = do_cross_validation();
	}
	else
	{
		if(*verbose)
			Rprintf("TRAIN\n");
		
		model_=train(&prob, &param);
		copy_model(W_ret, labels_ret, model_);
		free_and_destroy_model(&model_);
	}
	if(*verbose)
		Rprintf("FREE SPACE\n");
	
	//No need to destroy param because its members are shallow copies of Wi and WiLabels
	//destroy_param(&param);
	Free(prob.y);
	Free(prob.x);
	Free(x_space);
	
	if(*verbose)
		Rprintf("FREED SPACE\n");
	

	return;
}

/**
 * Function: do_cross_validation
 *
 * Author: Thibault Helleputte
 *
 */
double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);
	double res;
	
	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		res=total_error/prob.l;
		//squared_correlation_coefficient = 
		//		((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
		//		((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		res = 1.0*total_correct/prob.l;
	}
	
	Free(target);
	return(res);
}

/**
 * Function: setup_params
 *     Replaces parse_command_line from train.c
 * Author: Pierre Gramme
 *
 */
void setup_params(int *type, double *cost, double *epsilon, double* svr_eps, int *nrWi, double* C_vec, double *Wi, int *WiLabels, int *cross, int *verbose)
{
	if(*verbose){
		Rprintf("ARGUMENTS SETUP\n");
	}

	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// ARGUMENTS SETUP
	param.solver_type = *type;
	param.C = *cost;
	param.p = *svr_eps;
	param.eps = *epsilon;
	//Note: bias will be set in setup_problem(...)
	param.nr_weight = *nrWi;
	//TODO: deep copy might be safer than pointer copy
	param.W_bias = C_vec;
	param.weight_label = WiLabels;
	param.weight = Wi;
	
	if(*cross>0)
	{
		flag_cross_validation = 1; 
		nr_fold = *cross;
	}
	else
	{
		flag_cross_validation = 0; 
		nr_fold = 0;
	}

	// Verbose or not?
	if(!*verbose){
		// liblinear_print_string = &print_null;
		print_func = &print_null;
	}
	
	set_print_string_function(print_func);
	
	// NA value for eps is coded as <=0 instead of INF in original code
	//TODO in 1.94: update
	if(param.eps <= 0)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

/**
 * Function: setup_problem
 *     Replaces read_problem from train.c
 * Author: Pierre Gramme
 *
 */
void setup_problem(double *X, double *Y, int *nbSamples, int *nbDim, int *sparse, int *rowindex, int *colindex, 
                 double *bi, int *verbose)
{
	int i, j, k, max_index;
	i=j=k=0;
	
	if(*verbose){
		Rprintf("PROBLEM SETUP\n");
	}
	
	// PROBLEM SETUP
	prob.l = *nbSamples;
	prob.bias = *bi;
	
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	
	int allocSize = (*nbDim)*prob.l+prob.l;
	if (*sparse > 0){
		allocSize = rowindex[prob.l] + prob.l;
		if (*verbose)
			Rprintf("allocSize: %d\n",allocSize);
	}
	
	if(prob.bias >= 0)
		allocSize += prob.l;
		
	 x_space = Malloc(struct feature_node,allocSize);
	
	
	if(*verbose){
		Rprintf("FILL DATA STRUCTURE\n");
	}
	// Fill data stucture
	max_index = 0;
	k=0;
    if(*sparse > 0){
        // handle sparse matrix
        int totalK = 0;
        for(i=0; i<prob.l; i++){
            prob.y[i] = Y[i];
            prob.x[i] = &x_space[k];

            int nnz = rowindex[i+1]-rowindex[i];
            for(j=0; j<nnz; j++, k++, totalK++){
                x_space[k].index = colindex[totalK];
                x_space[k].value = X[totalK];

                if(colindex[totalK] > max_index){
                    max_index = colindex[totalK];
                }
            }

            if(prob.bias >= 0)
                x_space[k++].value = prob.bias;
            x_space[k++].index = -1;
        }
    }
    else {
        for(i=0;i<prob.l;i++){
            prob.y[i] = Y[i];
            prob.x[i] = &x_space[k];

            for(j=1;j<*nbDim+1;j++){
                if(X[(*nbDim*i)+(j-1)]!=0){
                    x_space[k].index = j;
                    x_space[k].value = X[(*nbDim*i)+(j-1)];
                    k++;
                    if(j>max_index){
                        max_index=j;
                    }
                }
            }
            if(prob.bias >= 0)
                x_space[k++].value = prob.bias;
            x_space[k++].index = -1;
        }
    }

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[k-2].index = prob.n;
	}
	else
		prob.n=max_index;
}


// give warning info if block information is incorrect
void warn_info (){
	std::cout << "Incorrect data blocks for its dimension!\n" << std::endl;
	std::cout << "Check bias setting might be helpful!\n" << std::endl;
	std::cout << "Program failed!\n" << std::endl;
}

// convert NumericMatrix to double array
void NM2array (NumericMatrix& NM, double *NM_array, bool byRow = false){
	int i,j,k;
	int nRow = NM.nrow();
	int nCol = NM.ncol();
	k = 0;
	if(byRow){
		for(i = 0; i < nRow; i++){
			for(j = 0; j < nCol; j++){
				NM_array[k] = NM(i,j);
				k++;
			}
		}
	}else{
		for(i = 0; i < nCol; i++){
			for(j = 0; j < nRow; j++){
				NM_array[k] = NM(j,i);
				k++;
			}
		}
	}
}

// convert double array to Numeric Matrix
void array2NM (NumericMatrix& NM, double *NM_array, bool byRow = false){
	int i,j,k;
	int nRow = NM.nrow();
	int nCol = NM.ncol();
	k = 0;
	if(byRow){
		for(i = 0; i < nRow; i++){
			for(j = 0; j < nCol; j++){
				NM(i,j) = NM_array[k];
				k++;
			}
		}
	}else{
		for(i = 0; i < nCol; i++){
			for(j = 0; j < nRow; j++){
				NM(j,i) = NM_array[k];
				k++;
			}
		}
	}
}


NumericVector vec_rep(NumericVector& a, int b){
	int n = a.size();
	NumericVector c(n*b);
	int k = 0;
	for(int i = 0; i < b; i++){
		for(int j = 0; j < n; j++){
			c[k] = a[j];
			k++;
		}
	}
	return c;
}

double sum_sq(NumericVector& x){
	int n = x.size();
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += pow(x(i),2);
	}
	return sum;
}

NumericVector col_norm(NumericMatrix& x){
	int Ncol = x.ncol();
	NumericVector x_norm_col(Ncol);
	for(int i = 0; i < Ncol; i ++){
		NumericVector x_col = x(_,i);
		x_norm_col[i] = sum_sq(x_col);
	}
	return x_norm_col;
}

NumericVector sum_by_block(NumericVector& x, int Nblock){
	int n = x.size();
	int Nblock_p = n/Nblock;
	NumericVector block_sum(Nblock_p);
	for(int i = 0; i < n; i++){
		block_sum[i % Nblock_p] += x[i];
	}
	NumericVector x_block = vec_rep(block_sum, Nblock);
	return x_block;
}

NumericVector C_grpnorm(NumericMatrix& C, int Nblock){
	NumericVector shrinkage = col_norm(C);
	shrinkage = sum_by_block(shrinkage, Nblock);
	shrinkage = sqrt(shrinkage);
	return shrinkage;
}

NumericMatrix proc_C(NumericMatrix& C, int Nblock, double penalty){
	int i,j;
	int K = C.nrow();
	int rp = C.ncol();
	NumericVector shrinkage = C_grpnorm(C, Nblock);
	NumericVector shrinkage_div(rp);
	NumericVector shrinkage_factor(rp);
	for(i = 0; i < rp; i++){
		if(shrinkage[i] == 0){
			shrinkage_div[i] = 1;
		}else{
			shrinkage_div[i] = shrinkage[i];
		}
		if(shrinkage[i] - penalty < 0){
			shrinkage_factor[i] = 0;
		}else{
			shrinkage_factor[i] = shrinkage[i] - penalty;
		}
	}

	NumericMatrix new_C(K,rp);
	for(i = 0; i < rp; i++){
		for(j = 0; j < K; j++){
			new_C(j,i) = C(j,i) * shrinkage_factor[i] / shrinkage_div[i];
		}
	}
	return new_C;
}

// get M
void update_M(NumericMatrix& C, NumericMatrix& M, int Nblock, double penalty, bool bias){
	int K = C.nrow();
	int rp = C.ncol();
	if(bias){
		NumericMatrix C_no_bias = C(Range(0,K-1),Range(0,rp-2));
		NumericMatrix newC = proc_C(C_no_bias, Nblock, penalty);
		for(int i = 0; i < K; i++){
			for(int j = 0; j < rp-1; j++){
				M(i,j) = newC(i,j);
			}
			M(i,rp-1) = C(i,rp-1);
		}
		// NumericMatrix new_C = proc_C(C, )
	}else{
		NumericMatrix newC = proc_C(C, Nblock, penalty);
		for(int i = 0; i < K; i++){
			for(int j = 0; j < rp; j++){
				M(i,j) = newC(i,j);
			}
		}
	}
}

// result = A+B or A-B
void mat_plus(NumericMatrix& result, NumericMatrix& A, NumericMatrix& B, bool add = true){
	int nRow = A.nrow();
	int nCol = A.ncol();
	int add_int = add ? 1: -1;
	for(int i = 0; i < nRow; i++){
		for(int j = 0; j < nCol; j++){
			result(i,j) = A(i,j) + B(i,j)*add_int;
		}
	}
}

double calcFnorm2(NumericMatrix& A, NumericMatrix& B){
	int nRow = A.nrow();
	int nCol = A.ncol();
	double Fnorm = 0;
	for(int i = 0; i < nRow; i++){
		for(int j = 0; j < nCol; j++){
			Fnorm += pow(A(i,j) - B(i,j),2);
		}
	}
	return Fnorm;
}

double calcFnorm1(NumericMatrix& A){
	int nRow = A.nrow();
	int nCol = A.ncol();
	double Fnorm = 0;
	for(int i = 0; i < nRow; i++){
		for(int j = 0; j < nCol; j++){
			Fnorm += pow(A(i,j),2);
		}
	}
	return Fnorm;
}

// sourceCpp('/Users/yayazhai/Documents/Courses/BIOSTAT815/project/myCode/RCPP_MCSVM/src/RCPP_MCSVM2.cpp')

// [[Rcpp::export]]
List MCSVM_grpl2(NumericMatrix X, IntegerVector y, int K, double cost, int Nblock, double mu, double lambda, double eps_VS, double eps_CD, bool add_bias = true, bool contain_bias = false, bool verbose = false){
	//MCSVM_grpl2(X = input_x, y = input_y, K = 5, cost = 1, Nblock = 2, mu = 10, lambda = 0.1, eps_VS = 0.01, eps_CD = 0.01,  add_bias = TRUE, contain_bias = FALSE, verbose = TRUE)
	int i,j,k;
	
	// get basic info about data
	int n = X.nrow();
	int p = X.ncol();
	int nc_W;
	double b;

	if(add_bias && contain_bias){
		std::cout << "Data already contain the bias term!\n" << std::endl;
		add_bias = false;
	}
	bool bias = add_bias || contain_bias; // whether the model contains a bias term in 

	if(add_bias){
		b = 1;
		nc_W = p + 1;
	}else{
		b = -1;
		nc_W = p;
	}

	// under bias case, nc_W = rp +1; otherwise, nc_W = rp
	if(bias){
		if((nc_W - 1) % Nblock != 0){
			warn_info();
			return List::create(Named("Run") = 0);
		}
	}else{
		if(nc_W % Nblock != 0){
			warn_info();
			return List::create(Named("Run") = 0);
		}
	}

	// initialize W, M, D, C
	int nr_W = K;
	NumericMatrix W(nr_W,nc_W);
	NumericMatrix M(nr_W,nc_W);
	NumericMatrix D(nr_W,nc_W);
	NumericMatrix C(nr_W,nc_W);

	// get the array version
	double W_ret[nr_W*nc_W];
	for(i = 0; i < nr_W*nc_W; i++){
		W_ret[i] = 0;
	}
	double C_vec[nr_W*nc_W];

	// prepare parameters to be passed to MCSVM
	// prepare data and target
	double Xt[n*p];
	k = 0;
	for(i = 0; i < n; i++){
		for(j = 0; j < p; j++){
			Xt[k] = X(i,j);
			k++;
		}
	}
	double yC[n];
	for(i = 0; i < n; i++){
		yC[i] = y(i);
	}
	// other parameters for MCSVM
	int labels_ret[K];
	for(i = 0; i < K; i++){
		labels_ret[i] = 0;
	}
	int sparse_int = 0;
	int type = 4;
	int nrWi = K;
	double Wi[nrWi];
	for(i = 0; i < nrWi; i++){
		Wi[i] = 1.0;
	}
	int WiLabels[K];
	for(i = 0; i < K; i++){
		WiLabels[i] = i+1;
	}
	int cross = 0;
	
	int rowindex = 0;
	int colindex = 0;
	double svr_eps = 0.1;
	int verbose_int = (int)verbose;

	double penalty = lambda/mu;

	int max_iter = 1000;
	int iter = 0;
	double Fnorm = 0;
	double Mnorm = 0;
	while(iter < max_iter){
		// variable splitting algorithm
		// ============================================================
		// ============= Start step 1, coordinate descent =============
		// solve W by coordinate descent method, adapted from in LiblineaR package
		// C = M + D;
		mat_plus(C, M, D, true);
		NM2array(C, C_vec, false); //convert to array

		// try get estimation of W using library
		trainLinear(W_ret, labels_ret, Xt, yC, &n, &p, C_vec, &sparse_int, &rowindex,  &colindex, 
	                  &b, &type, &cost, &eps_CD, &svr_eps, &nrWi, Wi, WiLabels, &cross, &verbose_int);
		
		// get coefficient matrix
		array2NM(W, W_ret, false);

		// ============ End of step 1, coordinate descent ============= 

		// ========= Start step 2, closed form solution for M ========= 
		// C = W - D;
		mat_plus(C,W,D,false);
		update_M(C, M, Nblock, penalty, bias);
		// ========= End of step 2, closed form solution of M ========= 
		// D = D - W + M; // Because C = W-D, D = M-C;
		mat_plus(D,M,C,false);
		// end of variable splitting
		// ============================================================
		iter++;
		Fnorm = calcFnorm2(W,M);
		Mnorm = calcFnorm1(M);
		if(Fnorm/Mnorm < eps_VS){
			break;
		}
	}

	return List::create(Named("W") = W, Named("C") = C, Named("M") = M, Named("D") = D, Named("Niter") = iter);
}
