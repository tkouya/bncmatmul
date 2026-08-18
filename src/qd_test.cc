/*#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "qd/dd_real.h"
#include "qd/qd_real.h"
#include "mpfr.h"
#include "bnc.h"
*/
#include "ddlinear.h"

// initialize qdvector
QDVector init_qdvector(long int dim)
{
	long int index;
	QDVector ret = NULL;

	// callocation
	ret = (qdvector *)malloc(sizeof(qdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one QDVector\n");
		return ret;
	}

	ret->dim = dim;
#ifdef __cplusplus
	ret->element = (qd_real *)calloc(dim, sizeof(qd_real));
//	ret->element = new qd_real[dim];
#else // __cplusplus
	ret->element = (double *)calloc(dim, sizeof(double) * QDSIZE);
#endif // __cplusplus
	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate QDVector(dim = %ld)\n", dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
	{
#ifdef __cplusplus
//		ret->element[index] = (qd_real)0.0;
		ret->element[index] = (qd_real)(int)(index + 1);
		ret->element[index] = sqrt(ret->element[index]);
#else // __cplusplus
		*(ret->element + index * QDSIZE)     = 0.0;
		*(ret->element + index * QDSIZE + 1) = 0.0;
		*(ret->element + index * QDSIZE + 2) = 0.0;
		*(ret->element + index * QDSIZE + 3) = 0.0;
#endif // __cplusplus
	}

	return ret;
}

using namespace std;

int main()
{
	mpfr_t mpfr_a;
	int i;
	dd_real dd_a, dd_b, dd_c;
	dd_real *dd_array;
	DDVector ddvec;
	qd_real a, b, c;
	qd_real *array;
	QDVector qdvec;

	fpu_fix_start(NULL);

	qdvec = init_qdvector(5);
	array = (qd_real *)calloc(5, sizeof(qd_real));
	//array = new qd_real[5];

	//a = sqrt(3.0);
	a = (qd_real)2.0;
	a = sqrt(a);
	b = sqrt(2.0);
	for(i = 0; i < 5; i++)
	{
		array[i] = (qd_real)(i + 1);
		array[i] = sqrt(array[i]);
		qdvec->element[i] = sqrt((qd_real)(i + 1));
	}

	cout.precision(qd_real::_ndigits);
	cout << "a         = " << a << endl;
	cout << "b         = " << b << endl;

	mpfr_set_default_prec(211);
	mpfr_init_set_ui(mpfr_a, 2UL, MPFR_RNDN);
	mpfr_sqrt(mpfr_a, mpfr_a, MPFR_RNDN);
	printf("sqrt(2.0) = "); mpfr_out_str(stdout, 10, 0, mpfr_a, MPFR_RNDN); printf("\n");
	mpfr_clear(mpfr_a);

	for(i = 0; i < 5; i++)
		cout << i << ": " << array[i] << endl;

	for(i = 0; i < 5; i++)
		cout << i << ": " << qdvec->element[i] << endl;

	free(array);
	//delete array;

	return 0;
}
