#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <float.h>
#include <assert.h>
#include "portable_time.h"

#include "portdefs.h"
#include "extern/matchmaker.h"


//#define PRINT_INFO	

// static double u_wseconds(void) 
// {
// 	struct timeval tp;

// 	gettimeofday(&tp, NULL);
// }

double dmpermTime, prmArrayfillTime;

static int minheap_insert(int *sz, int cap, double *vals, double el)
{
	int success = 0;

	if ( *sz == cap )
	{
	/*if minheap >= el, do nothing; 
	else  insert el at the 1st position and heapify there*/
		if (vals[1] >= el)
			success = 0;
		else
		{ 
			/*heapify from 1.*/
			int j = 2;
			while (j <= *sz)
			{
				if (j < *sz && vals[j] > vals[j+1] )
					j++;
				if (vals[j]>=el)
					break;
				else
				{
					vals[j>>1] = vals[j];
					j = j << 1;
				}

			}
			vals[j>>1] = el;
			success = 1;
		}
	}
	else /*mea,ing ( *sz < cap )*/
	{
		/*just insert*/
		int i = *sz = *sz + 1;
		while ( i > 1 && vals[i>>1] > el)
		{
			vals[i] = vals[i>>1];
			i = i >> 1;
		}

		vals[i] = el;
		success = 1;
	}
	return success;
}

typedef struct SortItem
{
	double v;
	int rid;
}tSortItem;


/*compare only the values*/
int vcmp(const void* a, const void* b)
{
	if( ((tSortItem*)a)->v > ((tSortItem*)b)->v ) /*we will sort in descending order*/
	  return -1;
	else
		return 1;
}

void sortVals(int* ptrs, int* ids, double *vals, int n, int m)
{

	tSortItem *myArray ;
	myArray = (tSortItem*) malloc(m * sizeof(tSortItem));
	int i, j;

	for (j = 0; j < n; j++)
	{
		int off = 0;
		int sz = ptrs[j+1] - ptrs[j];

		if (sz <= 1 ) continue;

		for (i = ptrs[j], off = 0; i < ptrs[j+1]; i++, off++)/*build the array to sort*/
		{
			myArray[off].v = vals[i];
			myArray[off].rid = ids[i];
		}	
		
		qsort(myArray, sz, sizeof(tSortItem), vcmp);

		for (i = ptrs[j], off = 0; i < ptrs[j+1]; i++, off++)/*copy the sorted array into output*/
		{
			vals[i] = myArray[off].v;
			ids[i] = myArray[off].rid;
		}
	}

	free(myArray);
}

/*from numerical recipes in C*/
#define RECSTACK 52
#define INSSWITCH 16

#define SWAPD(a, b) {tmpd = (a); (a) = (b); (b) = tmpd;}
#define SWAPI(a, b) {tmpi = (a); (a) = (b); (b) = tmpi;}

void myqsort(double *vals, int *ids, int sz)
{
	int i, ir = sz,j, k, l=1, anid, tmpi;
	int jstack = 0, istack[RECSTACK];
	double aval, tmpd;

/*as the following is a 1-based implementation, we adjust arrays first*/
	--vals;
	--ids;
	for (;;)
	{
		if(ir-l < INSSWITCH)
		{
			for(j = l+1; j <= ir; j++)
			{
				aval = vals[j];
				anid = ids[j];
				for (i = j-1; i >=1; i--)
				{
					if(vals[i] >= aval )
						break;
					vals[i+1] = vals[i];
					ids[i+1] = ids[i];
				}	
				vals[i+1] = aval;
				ids[i+1] = anid;
			}
			if(jstack == 0)
				break;
			ir = istack[jstack --];
			l = istack[jstack --];
		}
		else
		{
			k = (l+ir)>>1;
			/*swap k and l+1*/
			SWAPD(vals[k], vals[l+1]);
			SWAPI(ids[k], ids[l+1])
			if (vals[l+1] < vals[ir])
			{
				SWAPD(vals[l+1], vals[ir]);
				SWAPI(ids[l+1], ids[ir]);
			}
			if (vals[l] < vals[ir])
			{
				SWAPD(vals[l], vals[ir]);
				SWAPI(ids[l], ids[ir]);
			}
			if (vals[l+1] < vals[l])
			{
				SWAPD(vals[l+1], vals[l]);
				SWAPI(ids[l+1], ids[l]);
			}
			i = l+1;
			j = ir;
			aval = vals[l];
			anid = ids[l];
			for(;;)
			{
				do i++; while (vals[i] > aval);
				do j--; while (vals[j] < aval);
				if(j < i) break;
				SWAPD(vals[i], vals[j]);
				SWAPI(ids[i], ids[j]);
			}
			vals[l] = vals[j];
			ids[l] = ids[j];

			vals[j] = aval;
			ids[j] = anid;
			jstack +=2;
			if(jstack > RECSTACK)
			{
				printf("qsort stack is too small\n");
				exit(1);
			}
			if( ir -i + 1 >= j-l)
			{
				istack[jstack] = ir;
				istack[jstack-1] = i;
				ir = j-1;
			}
			else
			{
				istack[jstack] = j-1;
				istack[jstack-1] = l;
				l = i;
			}
		}
	}
}


void sortValsInsSortQS(int* ptrs, int* ids, double *vals, int n, int m)
{
	int i, j;
	int k, id;
	double kv;
	for (j = 0; j < n; j++)
	{
		int sz = ptrs[j+1] - ptrs[j];

		if (sz >=2)
		{
			if (sz <= INSSWITCH)/*insertion sort*/
			{
				for (k=ptrs[j]+1; k < ptrs[j+1]; k++)
				{	
					kv = vals[k];
					id = ids[k];
					i = k-1;
					while (i >= ptrs[j] && vals[i] < kv)
					{
						vals[i+1] = vals[i];
						ids[i+1] = ids[i];
						i --;
					}
					vals[i+1] = kv;
					ids[i+1] = id;
				}
			}
			else/*quicksort*/
			{
				myqsort(&(vals[ptrs[j]]), &(ids[ptrs[j]]), sz);
			}		
		}
		/*else: nothing to sort if sz < 2*/		
	}
}


void pr_global_relabel_bttlnck(int* l_label, int* r_label, int* row_ptrs, int *row_ends, int* row_ids, int* match, int* row_match, int n, int m) {
	int* queue = (int*)malloc(sizeof(int) * m);
	int relabel_vertex;
	int i;
	int	queue_end=-1;
	int	queue_start=0;

	int max = n+m;

	for(i=0; i <n; i++) 
		l_label[i]=max;

	for(i=0; i < m; i++) 
	{
		if (row_match[i] == -1) 
		{
			queue_end++;
			queue[queue_end] = i;
			r_label[i]=0;
		}
		else 
		{
			r_label[i]=max;
		}
	}

	while (queue_end-queue_start>=0) 
	{
		relabel_vertex=queue[queue_start];
		queue_start++;

		int ptr;
		int s_ptr = row_ptrs[relabel_vertex];
		int e_ptr = row_ends[relabel_vertex] + 1;
		for(ptr = s_ptr; ptr < e_ptr; ptr++) 
		{
			int left_vertex = row_ids[ptr];
			if(l_label[left_vertex] == max) 
			{
				l_label[left_vertex]=r_label[relabel_vertex]+1;
				if (match[left_vertex]> -1) 
				{
					if (r_label[match[left_vertex]] == max) 
					{
						queue_end++;
						queue[queue_end]=match[left_vertex];
						r_label[match[left_vertex]]=l_label[left_vertex]+1;
					}
				}
			}
		}
	}
	free(queue);
}

int match_pr_fifo_fair_bttlnck(int* col_ptrs, int *col_ends, int* col_ids, int* row_ptrs, int *row_ends, int* row_ids, int* match, int* row_match, int n, int m, double relabel_period) {
	int* l_label = (int*)malloc(sizeof(int) * n);
	int* r_label = (int*)malloc(sizeof(int) * m);

	int* queue = (int*)malloc(sizeof(int) * n);
	int	queue_end = -1;
	int	queue_start = 0;

	int max = m + n;
	int limit = (int)(max*relabel_period);
	if (relabel_period == -1) limit = m;
	if (relabel_period == -2) limit = n;

	int i = 0;
	int maxcmatching = 0;
	for(; i < n; i++) 
	{
		if (match[i] >= 0)
			maxcmatching++;
		else
		{
			queue_end++;
			queue[queue_end]=i;
		}
	}
	pr_global_relabel_bttlnck(l_label, r_label, row_ptrs, row_ends, row_ids, match, row_match, n, m);

	int min_vertex,max_vertex,min_label,next_vertex;
	int relabels=0;
	int queuesize = queue_end+1;

	while (queuesize>0) 
	{
		max_vertex=queue[queue_start];
		queue_start = (queue_start+1)%n;
		queuesize--;

		if (relabels==limit) 
		{
			pr_global_relabel_bttlnck(l_label, r_label, row_ptrs, row_ends, row_ids, match, row_match, n, m);
			relabels=0;
		}

		min_label=max;
		relabels++;

		if (l_label[max_vertex]<max) 
		{
			int ptr;
			int s_ptr = col_ptrs[max_vertex];
			int e_ptr = col_ends[max_vertex] + 1;
			if (l_label[max_vertex]%4==1) {
				for(ptr = s_ptr; ptr < e_ptr; ptr++) 
				{
					if(r_label[col_ids[ptr]] < min_label) 
					{
						min_label=r_label[col_ids[ptr]];
						min_vertex=col_ids[ptr];
						if (r_label[min_vertex]==l_label[max_vertex]-1)
						{
							relabels--;
							break;
						}
					}
				}
			} else {
				for(ptr = e_ptr-1; ptr >= s_ptr; ptr--) 
				{
					if(r_label[col_ids[ptr]] < min_label) 
					{
						min_label=r_label[col_ids[ptr]];
						min_vertex=col_ids[ptr];
						if (r_label[min_vertex]==l_label[max_vertex]-1)
						{
							relabels--;
							break;
						}
					}
				}
			}
		}

		if (min_label<max) 
		{
			if (row_match[min_vertex]==-1)
			{
				row_match[min_vertex]=max_vertex;
				match[max_vertex]=min_vertex;
				maxcmatching++;
			} 
			else 
			{
				next_vertex=row_match[min_vertex];
				queue_end = (queue_end+1)%n;
				queuesize++;
				queue[queue_end]=next_vertex;

				row_match[min_vertex]=max_vertex;
				match[max_vertex]=min_vertex;
				match[next_vertex]=-1;
				l_label[max_vertex]=min_label+1;
			}
			r_label[min_vertex]=min_label+2;
		}
	}

	free(queue);
	free(l_label);
	free(r_label);
	return maxcmatching;
}

/*
* marks vertices reachable from v with a BFS-like search
*/
static void bfsMark(int v, int *ptrs, int *eptrs, int *ids,  int *omatch, int type, int *whichtypeV, int *whichtypeOthers, int *vblock, int *oblock, int *queue, int n)
{
	int	qtop = -1, c, i, r;

	queue[++qtop] = v;
	whichtypeV[v] = type;

	while (qtop > -1)
	{
		c = queue[qtop--];
		(*vblock) ++;
		for (i = ptrs[c]; i <= eptrs[c]; i++)
		{
			r = ids[i];
			if(whichtypeV[omatch[r]] != type)
			{
				queue[++qtop] = omatch[r];
				whichtypeV[omatch[r]] = type;
				whichtypeOthers[r] = type;
				(*oblock) ++;
			}
		}				
	}
}

/*
* obtains coarse block decomposition. 
*
* returns rprm and cprm; normally rprm and cprm are ordered within the blocks so that
          matching nonzeros are along the diagonal. We do NOT provide this.
*
* returns rblks and cblks are allocated before and of size 4, 
*          they are filled with blocks' start addresses in rprm and cprm.
*
*/
void bdmperm(int *col_ptrs, int *col_eptrs, int *col_ids, int *row_ptrs, int *rows_eptrs, int *row_ids,
	int* match, int* row_match, int n, int m,
	int *rprm, int *cprm, int *rblks, int *cblks,
	int *whichtypeCol, int *whichtypeRow)
{
	int i, j, pos, type;
	int maxmn = m > n ? m : n;
	int *queue = malloc(sizeof(int) * maxmn);
	double t0, t1;
	memset(whichtypeCol, 0, sizeof(int) * n);/*this means square block*/

	memset(whichtypeRow, 0, sizeof(int) * m);/*this means square block*/

/* We use the followinf definitions for the DMPERM blocks (coarse)
  -1 horizontal rows/cols
   0 square rows/cols
   1 vertical rows/cols
 */
	rblks[0] = rblks[1] = rblks[2] = rblks[3] = 0; 
	cblks[0] = cblks[1] = cblks[2] = cblks[3] = 0; 

	/*1: find horizontal rows/cols*/
	for (j = 0; j < n; j++)
	{
		if(match[j] == -1 && whichtypeCol[j] != -1)/*unmatched column and was not visited before*/
		{
			/*start a bfs from j*/
			bfsMark(j, col_ptrs, col_eptrs, col_ids, row_match, -1, whichtypeCol, whichtypeRow, &(cblks[0]), &(rblks[0]), queue, n);
		}
	}

	/*2: find vertical rows/cols*/
	for (i = 0; i < m; i++)
	{
		if(row_match[i] == -1 && whichtypeRow[i] != 1)/*unmatched row and was not visited before*/
		{
			/*start a bfs from i*/
			bfsMark(i, row_ptrs, rows_eptrs, row_ids, match, 1, whichtypeRow, whichtypeCol, &(rblks[2]), &(cblks[2]), queue, m);
		}
	}
	/*3: the rest are square rows/cols*/

	cblks[1] = n - cblks[0] - cblks[2];
	rblks[1] = m - rblks[0] - rblks[2];

	/*prefix sum block starts*/
	for (i=1; i <= 3; i++)
	{
		cblks[i] += cblks[i-1];
		rblks[i] += rblks[i-1];
	}
	/*4: write cprm, rprm and adjust the block starts*/
	t0 = u_wseconds();
	for (j = n-1; j >= 0; j--)
	{
		type = whichtypeCol[j] + 1;
		pos = --cblks[type];
		cprm[pos] = j;
	}
	for (i = m-1; i >= 0; i--)
	{
		type = whichtypeRow[i] + 1;
		pos = --rblks[type];
		rprm[pos] = i;
	}
	t1 = u_wseconds();
	prmArrayfillTime += t1 - t0;

	free(queue);
}

void printMatrix(int *col_ptrs, int *fend_cols, int *col_ids, double *col_vals, int *row_ptrs, int *fend_rows, int *row_ids, double *row_vals, int n, int m) 
{
	int i, j;
	if (m > 400.0/n)
	{
		return;/*do nothing for large*/
	}	
	double *myA = (double*)malloc(m * n * sizeof(double));
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
			myA[i * n + j] = 0.0;
	}
	for (j= 0; j < n; j++)
	{
		for (int z = col_ptrs[j]; z <= fend_cols[j]; z++)
		{
			myA[col_ids[z] * n + j] = col_vals[z];
		}
	}
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			myPrintf(" %.4f", myA[i * n + j]);
		}
		myPrintf("\n");
	}
	myPrintf("=========Other view==========\n");
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
			myA[i * n + j] = 0.0;
	}
	for (i= 0; i < m; i++)
	{
		for (int z = row_ptrs[i]; z <= fend_rows[i]; z++)
		{
			myA[i * n + row_ids[z]] = row_vals[z];
		}
	}
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			myPrintf(" %.4f", myA[i * n + j]);
		}
		myPrintf("\n");
	}
	free(myA);
}
void minheapverify(int qsz, double *vals)
{
	int i;
	for (i = 1; i < qsz; i++)
	{
		double myv = vals[i];
		int l = i << 1;
		if (l <= qsz)
		{
			double lv = vals[l];
			if (lv < myv)
			{
				myPrintf("%.8f %.8f %d and %d \n", lv, myv, i, l);
				myExit("bottleneckBipartiteMatching.c: min heap not ok-1\n");
			}			
		}
		l = l+1;
		if (l <= qsz)
		{
			double lv = vals[l];
			if (lv < myv)
			{
				myPrintf("%.8f %.8f %d and %d \n", lv, myv, i, l);
				myExit("bottleneckBipartiteMatching.c: min heap not ok-2\n");
			}			
		}
	}
}
void maxheapverify(int qsz, int *myq, double *vals, int *posinq)
{
	int i,j ;
	/*check pos*/
	for(i = 1; i <= qsz; i++)
	{
		int r = myq[i];
		if(posinq[r] != i)
		{
			myPrintf("bottleneckBipartiteMatching.c: heap verify pos not ok %d vs %d sz %d\n", posinq[r], i, qsz );
			myExit("bottleneckBipartiteMatching.c: heap verify pos not ok\n");
		}
		if(posinq[r]>qsz)
			myExit("bottleneckBipartiteMatching.c: heap verify pos large\n");

		j = i<<1;
		if( j <= qsz )
		{
			if(vals[r] < vals[myq[j]])			
				myExit("bottleneckBipartiteMatching.c: heap verify val not ok 1\n");
			j++;
			if( j <= qsz)
			{
				if(vals[r] < vals[myq[j]])			
					myExit("bottleneckBipartiteMatching.c: heap verify val not ok 2\n");
			}
		}
	}
}
void maxheap_insert(int *qsz, int *myq, double *vals,  int *posinq, int id)
{
	int i, j;
	double dval = vals[id];
	j = ++(*qsz);
	i = j >> 1;
	while  (i > 0 && vals[myq[i]] < dval)
	{
		myq[j] = myq[i];
		posinq[myq[i]] = j;
		j = i;		
		i = i>>1;		
	}
	myq[j] = id;
	posinq[id] = j;
}

int maxheap_increaseKey(int qsz, int *myq, double *vals, int *posinq, int id, double nval)
{
	int pos = posinq[id];

	vals[id] = nval;
	int j = pos;
	int i = j >> 1;
	while(i > 0 && vals[myq[i]]<nval)
	{
		myq[j] = myq[i];
		posinq[myq[i]] = j;
		j = i;
		i = j>>1;	
	}
	myq[j] = id;
	posinq[id] = j;
	return j;
}
int maxheap_extract(int *qsz, int *myq, double *vals, int *posinq)
{
	int rid = myq[1];
	posinq[rid] = -1;

	if(*qsz == 1)
		*qsz = 0;
	else
	{	
		int id = myq[1] = myq[*qsz];
		double el = vals[id];
		posinq[id] = 1;
		int cap = --(*qsz);
		/****/
		int j = 2;/*heapify from 1 to cap*/
		while (j <= cap)
		{
			if (j < cap && vals[myq[j]] < vals[myq[j+1]] )
				j++;
			if (vals[myq[j]]<=el)
				break;
			else
			{
				myq[j>>1] = myq[j];
				posinq[myq[j]] = j>>1;
				j = j << 1;
			}

		}
		myq[j>>1] = id;
		posinq[id] = j>>1;
		/****/
	}
	return rid;
}


/* 
* Finds a shortest (in bottleneck terms) augmenting path starting from an unmatched column.
* returns the updated threshold
*/
int bmWithDijkstra(int *col_ptrs, int *col_eptrs, int *col_ids, double *col_vals,
	int* match, int* row_match, int n, int m,
	int *cprm, int *cblks,
 double *thrshld)
{
	int c,i,j,k, r, cofr, rofc, bestr, bestumc ;
	double *dists = (double *) malloc(sizeof(double) * m);/*by id to c*/
	int *myq = (int *) malloc(sizeof(int) * (m+1)); /* a heap; will store rows*/
	int *prnt = (int*) malloc(sizeof(int) * m);/*by id, for rows in the path, keeps the col lead to them*/
	int *posinq = (int*) malloc(sizeof(int) * m);

	int qsz;
	double bdist;
	for (i = 0; i < m; i ++)
	{
	//	dists[i] = -1.0;
		prnt[i] = -1;
	//	posinq[i] = -1;
	}
	bdist = *thrshld;
	bestumc = -1;
	for (j = 0; j < cblks[1]; j++)/*a hreustic: select an unmatched col vertex, whose largest discarded edge is the mminimum*/
	{
		c = cprm[j];
		if (match[c] != -1) continue;
		if (col_eptrs[c]+1 < col_ptrs[c+1])
		{
			if(col_vals[col_eptrs[c]+1 ] < bdist)
			{
				bdist = col_vals[col_eptrs[c]+1 ] ;
				bestumc = c;
			}
		}
	}
	qsz = 0;
	if (bestumc != -1)/*if we found such a column like that above*/
	{
		
		for (k = col_ptrs[bestumc]; k < col_ptrs[bestumc+1]; k++)
		{
			r = col_ids[k];
			prnt[r] = bestumc;
			double myv = *thrshld>col_vals[k] ? col_vals[k] : *thrshld;
			dists[r] = myv;

			maxheap_insert(&qsz, myq, dists, posinq, r);/*no need to check; only one col*/
		}
	}	
	else/*we could not find such a clumn like that above. Pick the first unmatched col*/
	{
		for (j = 0; j < cblks[1]; j++)/*horizontal cols*/
		{
			c = cprm[j];
			if (match[c] != -1) continue;
			for (k = col_ptrs[c]; k < col_ptrs[c+1]; k++)
			{
				r = col_ids[k];
				double myv = *thrshld>col_vals[k] ? col_vals[k] : *thrshld;
				dists[r] = myv;
				prnt[r] = c;
				maxheap_insert(&qsz, myq, dists, posinq, r);/*no need to check; only one col*/
			}
			break;/*get only one column*/
		}
	}

	bdist = -1.0;
	bestr = -1;
	while(qsz > 0)
	{
		r = maxheap_extract(&qsz, myq, dists, posinq);

		if(row_match[r] != -1)
		{
			cofr = row_match[r];
			for (k = col_ptrs[cofr]; k < col_ptrs[cofr+1]; k++)
			{
				if (col_ids[k] != r)
				{
					double myv = dists[r] > col_vals[k] ? col_vals[k] : dists[r];
					if (prnt[col_ids[k]] == -1) 
					{
						dists[col_ids[k]] = myv;
						prnt[col_ids[k]] = cofr;
						maxheap_insert(&qsz, myq, dists, posinq, col_ids[k]);
					}
					else 
					{
						if (dists[col_ids[k]] < myv)
						{
							maxheap_increaseKey(qsz, myq, dists, posinq, col_ids[k], myv);
							prnt[col_ids[k]] = cofr;
						}
					}
				}	
			}
		}
		else
		{
			bestr = r;
			break;
		}
	}
	if (bestr != -1)
	{	
		bdist = dists[bestr];
	/*augment*/

		*thrshld = bdist;
		r = bestr;
		while (r != -1)
		{
			c = prnt[r];
			rofc = match[c];
			row_match[r] = c;
			match[c] = r;
			r = rofc;
		}		
	}
	free(posinq);
	free(prnt);
	free(myq);
	free(dists);
	return bestr;
}
/*
* does three things:
*    i) sort each col in a non-increasing order of values, the same for the rows
*   ii) init fend_rows to point before the first element in each row (similary for fend_cols)
*  iii) initialize the threshold value.
*
* on input maxcmatch is the maximum cardinality of the matching in the original bipartute graph (sprank of the matrix)
*/

void bttlThresholdInitializer(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int nz, 
	int *row_ptrs, int *row_ids, double *row_vals,
	int *fend_cols, int *fend_rows,
	double *thrshld_g, int maxcmatch)
{
	double thrshld = DBL_MAX;
	int i,j;
	double t0 = u_wseconds();
	transpose(col_ptrs, col_ids, col_vals, row_ptrs, row_ids,  row_vals, n,  m) ;
	double t1 = u_wseconds();

#ifdef PRINT_INFO	
	myPrintf("Transposed in %.2f\n", t1-t0);
#endif 

	t0 = u_wseconds();
	sortValsInsSortQS(col_ptrs, col_ids, col_vals, n, m);
	sortValsInsSortQS(row_ptrs, row_ids, row_vals, m, n);
	t1 = u_wseconds();

#ifdef PRINT_INFO	
	myPrintf("sorted in %.2f\n", t1-t0);
#endif

	/*first  thrshld is the minimum of max in each row, in each column. Assuming a col perfect matchibg*/
	for (i = 0; i < m; i++)
	{
		fend_rows[i] = row_ptrs[i]-1;
	}

	for (j = 0; j < n; j++)
	{
		fend_cols[j] = col_ptrs[j]-1;
	}

	if (maxcmatch == n)/*if  n= sprank, then we simply get the min of the max*/
	{
		for (j = 0; j < n; j++)
		{
			if (col_vals[col_ptrs[j]] <  thrshld)
				thrshld = col_vals[col_ptrs[j]] ;
		}
	}
	else/*if n> sprank, then we get the sprank-th largest if the max*/
	{
		double *myhvals = (double *) malloc(sizeof(double) * (maxcmatch+1));
		int myheapsz = 0;
		for (i = 0; i < n; i++)
		{
			if(col_ptrs[i+1] > col_ptrs[i])
				minheap_insert(&myheapsz, maxcmatch, myhvals, col_vals[col_ptrs[i]]);
		}
		
		if (myheapsz >= 1)
			thrshld = thrshld > myhvals[1] ? myhvals[1] : thrshld;
		free(myhvals);
	}

	if(maxcmatch == m)/*if m = sprank, then we simply get the min of the max*/
	{
			for (i = 0; i < m; i++)
			{
				if (row_vals[row_ptrs[i]] < thrshld)
					thrshld = row_vals[row_ptrs[i]]  ;
			}	
	}
	else/*if m >  sprank, then we get the sprank-th largest if the max*/ 
	{
		double *myhvals = (double *) malloc(sizeof(double) * (maxcmatch+1));
		int myheapsz = 0;		
		for (i = 0; i < m; i++)
		{
			if(row_ptrs[i+1] > row_ptrs[i])
				minheap_insert(&myheapsz, maxcmatch, myhvals, row_vals[row_ptrs[i]]);
		}
		if (myheapsz >= 1)
			thrshld = thrshld > myhvals[1] ? myhvals[1] : thrshld;

		free(myhvals);
	}
	
	*thrshld_g = thrshld;
}


	/*
	* This is written from the point of view of columns, but it is also called from the rows.
	*
	* double *myheap contains at least myheapcap space.
	* 
	*/
double bidFromOneSide(int startblock, int endblock, int *cprm, int *col_ptrs, int *col_ids, double *col_vals, int *fend_cols,
	double *myheap, int myheapcap, int *whichtypeRow, int t1, int t2)
{

	int myheapsz, i, j, r, c;
	double mybid ;

	myheapsz = 0;
	mybid = DBL_MAX;

	for (j = startblock; j < endblock; j++)/*horizontal cols*/
	{
		double cmax = -1.0;
		c = cprm[j];
		for (i = fend_cols[c]+1; i < col_ptrs[c+1]; i++)
		{	
			r = col_ids[i];
			
			if(whichtypeRow[r] == t1 || whichtypeRow[r] == t2)/*rows either in S or in V for example*/
			{
				cmax =  col_vals[i];/*we insert only one (which is the max available) per column into the heap*/
				break;
			}
		}	
		if (cmax>0) minheap_insert(&myheapsz, myheapcap, myheap, cmax);
	}
	if(myheapsz>0)
		mybid = myheap[1];

	return mybid;	
}

/******************************** bttlThreshold ******************************** 
*	
*	the parameteters *col_ptrs, int *col_ids, double *col_vals, int n, int m, describe the mxn matrix
*				
*	the paramters are alllocated			
*       int *match (size n), int *row_match (size m), 
*				int *row_ptrs (size m+1), 
*				int *row_ids (size nnz)
*				double *row_vals (size nnz)
*				int *fend_cols (size n), int *fend_rows (size m)
*
* int lbapAlone: if used withing BvN decomposition, then lbapalone should be 1.
* 	 otherwise 0 (this subroutine is called as a linear bottlenack assignemnt/matching subroutine).
*
* double *thrshld_g: the computed bottleneck value at the end 
* 
* int sprankknown: if sprankknown == 0 at the beginning it is computed by this subroutine,
* 	otherwise, it is aqssumed to be equal to the sprank.
*/
#ifdef _WIN32
__declspec(dllexport)
#endif
int bttlThreshold(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int *match, int *row_match, 
		int *row_ptrs, 
		int *row_ids,
		double *row_vals,
		int *fend_cols, int *fend_rows,
		int lbapAlone, double *thrshld_g , int sprankknown)
{
	int numIters = 0;


	int i, j, prevdef, tmpsprank ;
	double  thrshld, altThCols, altThCols2, altThRows, altThRows2;
	int *rprm, *cprm, *rblks, *cblks, *whichtypeCol, *whichtypeRow;
	int  myheapcap;
	double totalMatchTime = 0.0, totalAugMatchTime = 0.0;
	int maxmn  = m > n ?  m : n;
	int minmn  = m > n ?  n : m;
	int initrankfull = sprankknown == minmn ? 1 : 0;
	double 	*myheap = (double *) malloc(sizeof(double) * (maxmn+1));/*+1 for 1 based indexing into the heap, +*/
	double t0, t1;

	double saveInThrs = *thrshld_g;

	if (sprankknown == 0)
	{
		int *			tmpspace = (int *) malloc(sizeof(int) * (m+n+1));
		sprankknown =  sprank(col_ptrs, col_ids,  n, m, tmpspace);
		free(tmpspace);
	}

	if(lbapAlone)
	{	
		bttlThresholdInitializer(col_ptrs, col_ids, col_vals, n,  m, col_ptrs[n],
			row_ptrs, row_ids, row_vals,
			fend_cols, fend_rows, thrshld_g, sprankknown);
	}


if(saveInThrs<*thrshld_g)
	// *thrshld_g = saveInThrs;
	thrshld = *thrshld_g;
	for (j = 0; j < n; j++)
		match[j] = -1;

	for (i = 0; i < m; i++)			
		row_match[i] =  -1;
	if(*thrshld_g < 0.0)
	{	
		numIters = -11;
		free(myheap);
		return numIters;
	}
	rprm = (int *) malloc(sizeof(int)* m);
	cprm = (int *)malloc(sizeof(int)* n);
	rblks = (int *)malloc(sizeof(int)* 4);
	cblks = (int *)malloc(sizeof(int)* 4);
	whichtypeCol = (int *) malloc(sizeof(int)* n);
	whichtypeRow = (int *) malloc(sizeof(int)* m);
	prevdef = n;
	dmpermTime = prmArrayfillTime = 0.0;

	while(1)
	{
		/* new filtered matrix contains col_ptrs[j] to col_fends[j], inclusive.
		 * new filtered matrix contains row_ptrs[i] to row_fends[i], inclusive.
		*/
#ifdef PRINT_INFO
		int bnewedges = 0;
#endif
		for (j = 0; j < n; j++)
		{
			while (fend_cols[j]+1 < col_ptrs[j+1] && col_vals[fend_cols[j]+1] >=  thrshld)			
				fend_cols[j]++;			
#ifdef PRINT_INFO
			bnewedges += fend_cols[j] - col_ptrs[j]+1;
#endif
		}

		for (i = 0; i < m; i++)
		{
			while (fend_rows[i]+1 < row_ptrs[i+1] && row_vals[fend_rows[i]+1] >=  thrshld)			
				fend_rows[i]++;			
		}

		t0 = u_wseconds();
		tmpsprank = match_pr_fifo_fair_bttlnck(col_ptrs, fend_cols, col_ids, row_ptrs, fend_rows, row_ids, match, row_match, n, m, 1.0) ;
		t1 = u_wseconds();
		totalMatchTime += t1-t0;

		if(tmpsprank == sprankknown)
		{
			prevdef = 0; 
			numIters ++;
			break;
		}

		t0 = u_wseconds();
		bdmperm(col_ptrs, fend_cols, col_ids, row_ptrs, fend_rows, row_ids, match, row_match,  n, m, rprm, cprm, rblks, cblks, whichtypeCol, whichtypeRow);
		t1 = u_wseconds();
		dmpermTime += t1-t0;
#ifdef PRINT_INFO
		myPrintf("%d thrshld %.4f, def %d, numEdges %d\n",  numIters+1, thrshld, sprankknown - ( rblks[2] /*hrows srows*/+ cblks[3] - cblks[2]/*vcols*/), bnewedges);
#endif
		numIters ++;
		if (sprankknown - ( rblks[2] /*hrows srows*/+ cblks[3] - cblks[2]/*vcols*/) == 0)/*check def*/
		{
			prevdef = 0;
			break;/*we have found the bottleneck matching. One of the above conditions is enough for our purposes (in BvN decomposition)*/
		}
/*if sprank is not equal to the min of m, n, then always thresholding; otherwise occasionally SAP based algorithm.*/


		if ((sprankknown != minmn ) || (prevdef !=1 && prevdef != (sprankknown - ( rblks[2] /*hrows srows*/+ cblks[3] - cblks[2]/*vcols*/))))
		{
			prevdef = myheapcap = sprankknown - ( rblks[2] /*hrows + srows*/+ cblks[3] - cblks[2]/*vcols*/); 

		 /*1st block identified with DMPERM*/
			altThCols =  bidFromOneSide(0, cblks[1], cprm, col_ptrs, col_ids, col_vals,fend_cols,
						myheap, myheapcap, whichtypeRow, 0, 1);
			thrshld = altThCols;

			 altThCols2 =  bidFromOneSide(rblks[1], rblks[3], rprm, row_ptrs, row_ids, row_vals,fend_rows,
						myheap, myheapcap, whichtypeCol, -1, -1);
			if (altThCols2 < thrshld)
				thrshld = altThCols2;
			
			/*2nd block identified with DMPERM*/
			altThRows = bidFromOneSide(rblks[2], rblks[3], rprm, row_ptrs, row_ids, row_vals,fend_rows,
						myheap, myheapcap, whichtypeCol, -1,  0);
			if(altThRows<thrshld)
				thrshld = altThRows;

			 altThRows2 =  bidFromOneSide(0, cblks[2], cprm, col_ptrs, col_ids, col_vals,fend_cols,
						myheap, myheapcap, whichtypeRow, 1, 1);
			if (altThRows2 < thrshld)
				thrshld = altThRows2;

			if(thrshld == DBL_MAX || thrshld < 0.0)
			{
				numIters = -11;
				break;
			}
		}
		else
		{
			int raug  = -1;
			if(thrshld == DBL_MAX || thrshld < 0.0)
			{
				numIters = -11;
				break;
			}
			if(sprankknown == n)
			{
				t0 = u_wseconds();				
				raug = bmWithDijkstra(col_ptrs, fend_cols, col_ids, col_vals, match, row_match, n,  m,cprm, cblks, &thrshld);
				t1 = u_wseconds();
			}
			else /*(sprankknown == m, which is < n)*/
			{
				t0 = u_wseconds();				
				raug = bmWithDijkstra(row_ptrs, fend_rows, row_ids, row_vals, row_match, match,  m, n, rprm, rblks, &thrshld);
				t1 = u_wseconds();
			}
			totalAugMatchTime += t1-t0;
			if((raug == -1 && initrankfull) || thrshld < 0.0)
			{
				numIters = -11;
				break;
			}
			prevdef --;
		}
	}
#ifdef PRINT_INFO	
	myPrintf("total match time %.2f augmatchTime %.2f dmpermTime %.2f prmFillTime %.2f def %d numiters %d\n", totalMatchTime, totalAugMatchTime, dmpermTime, prmArrayfillTime, prevdef, numIters);
#endif

	free(myheap);
	free(whichtypeRow);
	free(whichtypeCol);
	free(cblks);
	free(rblks);
	free(cprm);
	free(rprm);

	*thrshld_g = thrshld;
	return numIters;
}


int sprank(int *col_ptrs, int *col_ids,  int n, int m, int *tmpspace)
{
	int i, cardm = 0;
	int *match = tmpspace, *row_match = tmpspace + n;
	int matchID = 10, cheapID = 2;
	double relabel_period = 1.0;
/*
Algorithm definitions from matchmaker

#define do_old_cheap     1
#define do_sk_cheap      2
#define do_sk_cheap_rand 3
#define do_mind_cheap    4
#define do_truncrw       5

#define do_dfs 1
#define do_bfs 2
#define do_mc21 3
#define do_pf 4
#define do_pf_fair 5
#define do_hk 6
#define do_hk_dw 7
#define do_abmp 8
#define do_abmp_bfs 9
#define do_pr_fifo_fair 10
*/
	for (i = 0; i < m; i++) 
		row_match[i] = -1;

/*this is from MatchMaker*/
	matching(col_ptrs, col_ids, match, row_match, n, m, matchID, cheapID, relabel_period);

	for( i = 0; i < m; i ++)
	{
		if(row_match[i] >= 0) 
			cardm ++;
	}

	return cardm;
}

int cmpfunc (const void * a, const void * b) 
{
	if (*(double*)a > *(double*)b) 
		return 1;
	else
		return -1;
}

void filterMatrix(int *col_ptrs, int *col_ids, double *col_vals, int n,  double v,
				int *fcol_ptrs, int *fcol_ids)
{
	int i, j, off;

	fcol_ptrs[0] = 0;
	for (i = 0; i < n; i++)
	{
		off = fcol_ptrs[i];
		for (j = col_ptrs[i] ; j < col_ptrs[i+1]; j++)
		{
			if(col_vals[j]>=v)
			{
				fcol_ids[off++] = col_ids[j];
			}
		}
		fcol_ptrs[i+1] = off;
	}
}
/*
*
*A helper routine for the bisection based algorithm.
*For each row, fend pointers are adjusted according to the current threshold either gets 
*      larger (threshoold reduced) or smaller (threshold increased). In the second case,
*      matching edge becomes free as we do not know if it is available next. 
*The same for the columns.
*/
void updateAfterBisection(int *fend_cols, int *col_ptrs, int *col_ids , double *col_vals, double v, int *match, int *row_match, int n)
{
	int j;
	for (j = 0; j < n; j++)
	{
		while (fend_cols[j]+1 < col_ptrs[j+1] && col_vals[fend_cols[j]+1] >= v)	/*new entries not used in the current matching*/		
			fend_cols[j]++;			
		while (fend_cols[j] >= col_ptrs[j] && col_vals[fend_cols[j]]<v)/*discard entries, potentially used in the current matching*/		
		{
			if(match[j] == col_ids[fend_cols[j]] )
			{
				row_match[match[j] ] = -1;
				match[j] = -1;
			}
			fend_cols[j]--;
		}
	}
}

/*

we start with an initial extreme matching (with the min of max rows, max cols), then for each 
unmatched column vertex, we will run a SAP.

	the parameteters *col_ptrs, int *col_ids, double *col_vals, int n, int m, describe the mxn matrix
	We assume m>=n, and there is a column-perfect matching.

	the following parameters are allocated	by the caller function:
				int *match (size n), int *row_match (size m), 
				int *row_ptrs (size m+1), 
				int *row_ids (size nnz)
				double *row_vals (size nnz)
				int *fend_cols (size n), int *fend_rows (size m)

double *thrshld; computed the bottleneck value at the end 

int sprankknown: if sprankknown == 0 at the beginning it is computed by this subroutine,
	otherwise, it is assumed to be equal to the sprank.

*/
int pureSAP(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int *match, int *row_match, 
				int *row_ptrs, 
				int *row_ids,
				double *row_vals,
				int *fend_cols, int *fend_rows, double *thrshld, int sprankknown)
{

	int i, j, k, c, r, cofr, rofc, bestr;
	int cardm;
	double initval  ;


	if (sprankknown == 0)
	{
		int *	tmpspace = (int *) malloc(sizeof(int) * (m+n+1));
		sprankknown =  sprank(col_ptrs, col_ids,  n, m, tmpspace);
		free(tmpspace);
	}
	
	if(sprankknown != n)
	{
		myExit("no column perfect matching. Exiting from pureSAP.\n");
	}

	bttlThresholdInitializer(col_ptrs, col_ids, col_vals, n,  m, col_ptrs[n],
			row_ptrs, row_ids, row_vals,
			fend_cols, fend_rows, &initval, sprankknown);
	
	for (j = 0; j < n; j++)
		match[j] = -1;

	for (i = 0; i < m; i++)			
		row_match[i] =  -1;

	for (j = 0; j < n; j++)
	{
		while (fend_cols[j]+1 < col_ptrs[j+1] && col_vals[fend_cols[j]+1] >=  initval)			
			fend_cols[j]++;					
	}
	for (i = 0; i < m; i++)
	{
		while (fend_rows[i]+1 < row_ptrs[i+1] && row_vals[fend_rows[i]+1] >=  initval)			
			fend_rows[i]++;			
	}

	/*a first check with initval to initialize and potentially avoid the bisection*/
	cardm = match_pr_fifo_fair_bttlnck(col_ptrs, fend_cols, col_ids, row_ptrs, fend_rows, row_ids, match, row_match, n, m, 1.0) ;
	*thrshld = initval;

	if(cardm < sprankknown)
	{
		double *dists = (double *) malloc(sizeof(double) * m);/*by id to c*/
		int *myq = (int *) malloc(sizeof(int) * (m+1)); /* a heap; will store rows*/
		int *prnt = (int*) malloc(sizeof(int) * m);/*by id, for rows in the path, keeps the col lead to them*/
		int *posinq = (int*) malloc(sizeof(int) * m);
		int *seenthisround = (int*) malloc(sizeof(int) * m);

		int qsz;
		double bdist;
		for (j = 0; j < m; j++)
			seenthisround[j] = -1;

		for (j = 0; j < n; j++)
		{
			if(match[j] >= 0)
				continue;

			/*a free column vertex*/
			qsz = 0;
			for (k = col_ptrs[j]; k < col_ptrs[j+1]; k++)
			{
				r = col_ids[k];
				double myv = *thrshld>col_vals[k] ? col_vals[k] : *thrshld;
				dists[r] = myv;
				prnt[r] = j;
				seenthisround[r] = j;

				maxheap_insert(&qsz, myq, dists, posinq, r);/*no need to check; only one col*/
			}			
			bdist = -1.0;
			bestr = -1;
			while(qsz > 0)
			{
				r = maxheap_extract(&qsz, myq, dists, posinq);

				if(row_match[r] != -1)
				{
					cofr = row_match[r];
					for (k = col_ptrs[cofr]; k < col_ptrs[cofr+1]; k++)
					{
						if (col_ids[k] != r)
						{
							double myv = dists[r] > col_vals[k] ? col_vals[k] : dists[r];
							if (seenthisround[col_ids[k]] != j) 
							{
								seenthisround[col_ids[k]] = j;
								dists[col_ids[k]] = myv;
								prnt[col_ids[k]] = cofr;
								maxheap_insert(&qsz, myq, dists, posinq, col_ids[k]);
							}
							else 
							{
								if (dists[col_ids[k]] < myv)
								{
									maxheap_increaseKey(qsz, myq, dists, posinq, col_ids[k], myv);
									prnt[col_ids[k]] = cofr;
								}
							}
						}	
					}
				}
				else
				{
					bestr = r;
					break;
				}
			}
			if (bestr != -1)
			{	
				bdist = dists[bestr];
				/*augment*/
				*thrshld = bdist;
				r = bestr;
				while (r != -1)
				{
					c = prnt[r];
					rofc = match[c];
					row_match[r] = c;
					match[c] = r;
					r = rofc;
				}		
			}
			else
			{
				myExit("error. error. error. error. \ncould not match a column. Exiting from pureSAP.\n");
			}
		}/*for j = 0 to n..*/
		free(seenthisround);
		free(posinq);
		free(prnt);
		free(myq);	
		free(dists);
	}/*of if(cardm < sprankknown)*/

return sprankknown - cardm;
}

/* 
	the parameteters *col_ptrs, int *col_ids, double *col_vals, int n, int m, describe the mxn matrix
				
	the following parameters are allocated	by the caller function:
				int *match (size n), int *row_match (size m), 
				int *row_ptrs (size m+1), 
				int *row_ids (size nnz)
				double *row_vals (size nnz)
				int *fend_cols (size n), int *fend_rows (size m)

double *thrshld; computed the bottleneck value at the end 

int sprankknown: if sprankknown == 0 at the beginning it is computed by this subroutine,
	otherwise, it is assumed to be equal to the sprank.
	*/
int bisectionBasedOnMC64J3(int *col_ptrs, int *col_ids, double *col_vals, int n, int m, int *match, int *row_match, 
				int *row_ptrs, 
				int *row_ids,
				double *row_vals,
				int *fend_cols, int *fend_rows, double *thrshld, int sprankknown)
{

	int i, j, nnz = col_ptrs[n];
	int minind = 0, maxind = nnz; 
	int iters;	
	int cardm;
double t0, t1;
	double *tmpvals = (double*) malloc(sizeof	(double) * nnz);
	double initval  ;



	if (sprankknown == 0)
	{
		int *	tmpspace = (int *) malloc(sizeof(int) * (m+n+1));
		sprankknown =  sprank(col_ptrs, col_ids,  n, m, tmpspace);
		free(tmpspace);
	}
	
	bttlThresholdInitializer(col_ptrs, col_ids, col_vals, n,  m, col_ptrs[n],
			row_ptrs, row_ids, row_vals,
			fend_cols, fend_rows, &initval, sprankknown);
	
if(*thrshld < initval)
	initval = *thrshld;

	for (j = 0; j < n; j++)
		match[j] = -1;

	for (i = 0; i < m; i++)			
		row_match[i] =  -1;

	for (j = 0; j < n; j++)
	{
		while (fend_cols[j]+1 < col_ptrs[j+1] && col_vals[fend_cols[j]+1] >=  initval)			
			fend_cols[j]++;					
	}
	for (i = 0; i < m; i++)
	{
		while (fend_rows[i]+1 < row_ptrs[i+1] && row_vals[fend_rows[i]+1] >=  initval)			
			fend_rows[i]++;			
	}

	/*a first check with initval to initialize and potentially avoid the bisection*/
	cardm = match_pr_fifo_fair_bttlnck(col_ptrs, fend_cols, col_ids, row_ptrs, fend_rows, row_ids, match, row_match, n, m, 1.0) ;
	
myPrintf("sprank known %d\n", sprankknown);
	if(cardm < sprankknown)
	{
		t0 = u_wseconds();
		memcpy(tmpvals, col_vals, sizeof(double) * nnz);
		qsort(tmpvals, nnz, sizeof(double), cmpfunc);
		t1 = u_wseconds();
#ifdef PRINT_INFO
		printf("bisectionBasedOnMC64J3: initial sort for conducting the bisection %.2f\n", t1-t0);
#endif
		for (i = 0; i < nnz; i++)
		{
			if (tmpvals[i] > initval)
			{
				maxind = i;
				break;
			}	
		}
	}
	iters = 1;/*1 because has called matchi_pr_fifo_fair_bttlnck once allreads*/
	*thrshld = initval;

#ifdef PRINT_INFO
	myPrintf("initval %.4f at %d with sprank %d (wrt n %d) card %d\n", initval, maxind, sprankknown, n, cardm);
#endif
	/*if the initval did not give perfect matching, then search*/
	if(cardm < sprankknown)
	{
		while (minind < maxind-1)
		{
			if (tmpvals[minind] == tmpvals[maxind-1])/*if all values are the same in this range*/
			{
				*thrshld = tmpvals[minind];
  			/*In case we shortcut, we shortcut after computing a matching*/

				updateAfterBisection(fend_cols, col_ptrs, col_ids, col_vals,  tmpvals[minind], match, row_match, n);
				updateAfterBisection(fend_rows, row_ptrs, row_ids, row_vals,  tmpvals[minind], row_match, match, m);

				cardm = match_pr_fifo_fair_bttlnck(col_ptrs, fend_cols, col_ids, row_ptrs, fend_rows, row_ids, match, row_match, n, m, 1.0) ;
				break;
			}
			int mid = (minind + maxind)/2;
			double v = tmpvals[mid];
#ifdef PRINT_INFO
			printf("%d %d %d %.4f\n", iters, minind, maxind, v);
#endif

			updateAfterBisection(fend_cols, col_ptrs, col_ids, col_vals, v, match, row_match, n);
			updateAfterBisection(fend_rows, row_ptrs, row_ids, row_vals, v, row_match, match, m);

			cardm = match_pr_fifo_fair_bttlnck(col_ptrs, fend_cols, col_ids, row_ptrs, fend_rows, row_ids, match, row_match, n, m, 1.0) ;

			if(cardm < sprankknown)
				maxind = mid;
			else
			{
				minind = mid;
				*thrshld = tmpvals[mid];
			}
			iters ++;	
			if (minind == maxind-1)
			{
				*thrshld = tmpvals[minind];
  			/*In case we shortcut, we shortcut after computing a matching*/
				updateAfterBisection(fend_cols, col_ptrs, col_ids, col_vals,  tmpvals[minind], match, row_match, n);
				updateAfterBisection(fend_rows, row_ptrs, row_ids, row_vals,  tmpvals[minind], row_match, match, m);

				cardm = match_pr_fifo_fair_bttlnck(col_ptrs, fend_cols, col_ids, row_ptrs, fend_rows, row_ids, match, row_match, n, m, 1.0) ;
				break;
			}
		}
	}
#ifdef PRINT_INFO
	myPrintf("num E %d minind %d maxind %d numiters %d\n", nnz, minind, maxind, iters);
#endif
	free(tmpvals);

	return iters;
}

