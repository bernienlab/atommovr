#include <stdio.h>

#ifndef _PORTDEFS_H_
#define _PORTDEFS_H_

#define myPrintf printf

#define myExit(str) {\
						printf("%s\n", str); \
						exit(1);\
					}

// #ifdef MAIN_C
// #define myPrintf printf

// #define myExit(str) {\
// 						printf("%s\n", str); \
// 						exit(1);\
// 					}
// #else
// 	#include "mex.h"
// 	#define myPrintf mexPrintf
// 	#define myExit mexErrMsgTxt
// #endif 

/*utilities*/
void transpose(int* sptrs, int* sids, double *svals, int* tptrs, int* tids, double *tvals, int n, int m) ;
int sprank(int *col_ptrs, int *col_ids,  int n, int m, int *tmpspace);
void shuffle(int *a, int n);

/*different bottleneck matching and their initializer*/
void bttlThresholdInitializer(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int nz, 
	int *row_ptrs, int *row_ids, double *row_vals,
	int *fend_cols, int *fend_rows,
	double *thrshld_g, int maxcrdmatch);

#ifdef _WIN32
__declspec(dllexport)
#endif 
int bttlThreshold(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int *match, int *row_match, 
	int *row_ptrs, 
	int *row_ids,
	double *row_vals,
	int *fend_cols, int *fend_rows,
	int lbapAlone, double *thrshld_g, int sprankknown);

int bisectionBasedOnMC64J3(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int *match, int *row_match, 
	int *row_ptrs, 
	int *row_ids,
	double *row_vals,
	int *fend_cols, int *fend_rows, double *thrshld, int sprankknown);

int pureSAP(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int *match, int *row_match, 
				int *row_ptrs, 
				int *row_ids,
				double *row_vals,
				int *fend_cols, int *fend_rows, double *thrshld, int sprankknown);


#endif
