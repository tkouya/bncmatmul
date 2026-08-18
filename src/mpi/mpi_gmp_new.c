/**********************************************/
/* mpi_gmp.c:                                 */
/* Copyright (C) 2003 Tomonori Kouya          */
/*                                            */
/* Version 0.0: 2003.04/01                    */
/* Version 0.1: 2003.04/21                    */
/* Version 0.2: 2003.05/08 mpz_t, mpq_t       */
/* Version 0.3: 2003.05/26 modify _mpi_mpf_add */
/*                                            */
/* This library is free software; you can re- */
/* distribute it and/or modify it under the   */
/* terms of the GNU Lesser General Public     */
/* License as published by the Free Software  */
/* Foundation; either version 2.1 of the      */
/* License, or (at your option) any later     */
/* version.                                   */
/*                                            */
/* This library is distributed in the hope    */
/* that it will be useful, but WITHOUT ANY    */
/* WARRANTY; without even the implied         */
/* warranty of MERCHANTABILITY or FITNESS FOR */
/* A PARTICULAR PURPOSE.  See the GNU Lesser  */
/* General Public License for more details.   */
/**********************************************/
#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

/* divide number of dimension */
long int _mpi_divide_dim(long int d_dim[], long int dim, int num_procs)
{
	long int rem, sum, old_sum, i, local_dim;

	/* error = -1 */
	if((num_procs <= 0) || (dim <= 0))
		return -1;

	/* num_procs >= dim  -> local_dim = 1 */
	/* num_procs <  dim  -> local_dim > 1 */
	local_dim = dim / num_procs;
	rem = dim % num_procs;
	if(rem > 0)
		local_dim++;

	sum = 0;
	for(i = 0; i < num_procs; i++)
	{
		if(sum >= dim)
			d_dim[i] = 0;
		else
		{
			old_sum = sum;
			sum += local_dim;
			if(sum > dim)
			{
				sum = dim;
				d_dim[i] = dim - old_sum;
			}
			else
				d_dim[i] = local_dim;
		}

	}

	return local_dim;
}

#ifdef USE_GMP
/* get bufsize for mpz_t */
/* mpz_t variable is potentially re-allocated if needed */
size_t get_bufsize_mpz(mpz_t a)
{
	size_t bufsize;

	bufsize = sizeof(int) * 2 + sizeof(mp_limb_t) * a->_mp_alloc;

	return bufsize;
}

/* get bufsize for mpq_t */
/* mpq_t variable is potentially re-allocated if needed, too. */
size_t get_bufsize_mpq(mpq_t a)
{
	size_t bufsize;

	bufsize = get_bufsize_mpz(&(a->_mp_num)) + get_bufsize_mpz(&(a->_mp_den));

	return bufsize;
}


/* get bufsize for mpf_t */
size_t get_bufsize_mpf(mpf_t a, int incount)
{
	size_t bufsize;

#ifdef __MPFR_H
	/* Definition of the main structure */
	//
	//	typedef struct {
	//	  mpfr_prec_t  _mpfr_prec;
	//	  mpfr_sign_t  _mpfr_sign;
	//	  mp_exp_t     _mpfr_exp;
	//	  mp_limb_t   *_mpfr_d;
	//	} __mpfr_struct;
	//	typedef __mpfr_struct mpfr_t[1];
	//	typedef __mpfr_struct *mpfr_ptr;
	//	typedef __gmp_const __mpfr_struct *mpfr_srcptr;
	//	above def. from mpfr.h in mpfr-2.2.0
	//
	//bufsize = sizeof(mpfr_sign_t) + sizeof(mpfr_prec_t) + sizeof(mp_exp_t) + sizeof(mp_limb_t) * _NUM_LIMB(a);
	bufsize = sizeof(mpfr_sign_t) + sizeof(mpfr_prec_t) + sizeof(mp_exp_t);
	bufsize += sizeof(mp_limb_t) * ((mpfr_get_prec((mpfr_ptr)a) - 1) / GMP_LIMB_BITS + 1);
	//bufsize = sizeof(int) * 2 + sizeof(mp_exp_t) + sizeof(mp_limb_t) * _NUM_LIMB(a);
#else
	// typedef struct
	// {
	//		int _mp_prec;
	//		int _mp_size;
	//		mp_exp_t _mp_exp;		 Exponent, in the base of `mp_limb_t'.  */
	//		mp_limb_t *_mp_d;		 Pointer to the limbs.  */
	//	} __mpf_struct;
	// typedef __mpf_struct MP_FLOAT; 
	// typedef __mpf_struct mpf_t[1];
	// above def. from gmp.h in gmp-4.1.4
	//
	bufsize = sizeof(int) * 2 + sizeof(mp_exp_t) + sizeof(mp_limb_t) * (a->_mp_prec + 1);
#endif

	return (size_t)(bufsize * incount);
}

/* get bufsize for mpf_t and prec(unsigned long) */
size_t get_bufsize_mpf2(mpf_t a, int incount)
{
	size_t bufsize;

	bufsize = get_bufsize_mpf(a, incount) + sizeof(unsigned long);

	return bufsize;
}
	
/* allocate buf for mpz_t                     */
void *allocbuf_mpz(mpz_t a)
{
	void *buf;
	int bufsize;

	buf = (void *)malloc(get_bufsize_mpz(a));

	return buf;
}

/* allocate buf for mpq_t                     */
void *allocbuf_mpq(mpq_t a)
{
	void *buf;
	int bufsize;

	buf = (void *)malloc(get_bufsize_mpq(a));

	return buf;
}

/* allocate buf for mpf_t                     */
/* |<---------get_bufsize_mpf()------------>| */
/* +--------+--------+---+------------------+ */
/* |mpf_t[0]|mpf_t[1]|...|mpf_t[incount - 1]| */
/* +--------+--------+---+------------------+ */
void *allocbuf_mpf(unsigned long prec, int incount)
{
	void *buf;
	int bufsize;
	mpf_t a;

	mpf_init2(a, prec);
	
	buf = (void *)malloc(get_bufsize_mpf(a, incount));

	mpf_clear(a);

	return buf;
}

/* allocate buf for mpf_t and prec                 */
/* |<------------get_bufsize_mpf2()------------->| */
/* +----+--------+--------+---+------------------+ */
/* |prec|mpf_t[0]|mpf_t[1]|...|mpf_t[incount - 1]| */
/* +----+--------+--------+---+------------------+ */
void *allocbuf_mpf2(unsigned long prec, int incount)
{
	void *buf;
	int bufsize;
	mpf_t a;

	mpf_init2(a, prec);
	
	buf = (void *)malloc(get_bufsize_mpf2(a, incount));

	mpf_clear(a);

	return buf;
}

void freebuf_mpz(void *buf)
{
	free(buf);
}

void freebuf_mpq(void *buf)
{
	free(buf);
}

void freebuf_mpf(void *buf)
{
	free(buf);
}

void freebuf_mpf2(void *buf)
{
	free(buf);
}

/* pack mpz */
void pack_mpz(mpz_t a, void *buf)
{
	int i;
	unsigned char *tmp_buf;
	mpz_ptr ptr_a;

	if(buf == NULL)
	{
		buf = allocbuf_mpz(a);
	}

	tmp_buf = (unsigned char *)buf;
	ptr_a = (mpz_ptr)a;

/* 1. int a->_mp_alloc; */
	memcpy(tmp_buf, &(a->_mp_alloc), sizeof(int));
	tmp_buf += sizeof(int); ptr_a += sizeof(int);

/* 2. int a->_mp_size; */
	memcpy(tmp_buf, &(a->_mp_size), sizeof(int));
	tmp_buf += sizeof(int); ptr_a += sizeof(int);

/* 3. mp_limb_t(unsigned long?) a->_mp_d; */
	memcpy(tmp_buf, a->_mp_d, (size_t)(sizeof(mp_limb_t) * (a->_mp_alloc)));
}

/* unpack mpz */
void unpack_mpz(void *buf, mpz_t ret)
{
	int allocsize;
	unsigned long prec;
	unsigned char *tmp_buf;
	unsigned char *ptr_ret;
	mpz_ptr tmp_ret;

	/* get _mp_alloc */
	memcpy(&allocsize, buf, sizeof(int));

	/* realloc to fit received buf */
	_mpz_realloc(ret, allocsize);

	tmp_buf = (unsigned char *)buf;
	ptr_ret = (unsigned char *)ret;

/* 1. int ret->_mp_alloc; */
	memcpy(&(ret->_mp_alloc), tmp_buf, sizeof(int));
	tmp_buf += sizeof(int); ptr_ret += sizeof(int);

/* 2. int ret->_mp_size; */
	memcpy(&(ret->_mp_size), tmp_buf, sizeof(int));
	tmp_buf += sizeof(int); ptr_ret += sizeof(int);

/* 3. mp_limb_t ret->_mp_d; */
	memcpy(ret->_mp_d, tmp_buf, sizeof(mp_limb_t) * (ret->_mp_alloc));

}

/* pack mpq */
void pack_mpq(mpq_t a, void *buf)
{
	unsigned char *tmp_buf;

	if(buf == NULL)
	{
		buf = allocbuf_mpq(a);
	}

	tmp_buf = (unsigned char *)buf;

/* 1. _mpz_struct a->_mp_num; */
	pack_mpz(&(a->_mp_num), tmp_buf);
	tmp_buf += get_bufsize_mpz(&(a->_mp_num));

/* 2. _mpz_struct a->_mp_den; */
	pack_mpz(&(a->_mp_den), tmp_buf);

}

/* unpack mpq */
void unpack_mpq(void *buf, mpq_t ret)
{
	unsigned char *tmp_buf;

	tmp_buf = (unsigned char *)buf;

/* 1. _mpz_struct ret->_mp_num; */
	unpack_mpz(tmp_buf, &(ret->_mp_num));
	tmp_buf += get_bufsize_mpz(&(ret->_mp_num));

/* 2. _mpz_struct ret->_mp_den; */
	unpack_mpz(tmp_buf, &(ret->_mp_den));

}

/* pack mpf */
#ifdef __USE_MPI_PACK
void pack_mpf(mpf_t a, int incount, void *buf, int *bufsize, int *pos, MPI_Comm comm)
#else
void pack_mpf(mpf_t a, int incount, void *buf)
#endif
{
	int i;
	unsigned long prec;
	unsigned char *tmp_buf;
	unsigned char *ptr_a;
	mpf_ptr tmp_a;

	prec = mpf_get_prec(a);

	if(buf == NULL)
	{
		buf = allocbuf_mpf(prec, incount);
	}

	tmp_buf = (unsigned char *)buf;
	ptr_a = (unsigned char *)a;
	tmp_a = (mpf_ptr)a;

	for(i = 0; i < incount; i++)
	{
		// printf("pack_mpf->"); mpf_out_str(stdout, 10, 0, a); printf("\n");
/* 2. int a->_mp_size; */
/* 3. int a->_mp_prec; */
/* 4. mp_exp_t(long?) a->_mp_exp; */
/* 5. mp_limb_t(unsigned long?) a->_mp_limb; */

#ifndef __USE_MPI_PACK
#ifdef __MPFR_H
		memcpy(tmp_buf, &(a->_mpfr_prec), sizeof(mpfr_prec_t));
		tmp_buf += sizeof(mpfr_prec_t); ptr_a += sizeof(mpfr_prec_t);
#if MPFR_VERSION_MAJOR >= 2
#if MPFR_VERSION_MINOR >= 1
		memcpy(tmp_buf, &(a->_mpfr_sign), sizeof(mpfr_sign_t));
		tmp_buf += sizeof(mpfr_sign_t); ptr_a += sizeof(mpfr_sign_t);
#else
		memcpy(tmp_buf, &(a->_mpfr_size), sizeof(int));
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
#endif
#else
		memcpy(tmp_buf, &(a->_mpfr_size), sizeof(int));
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
#endif
		memcpy(tmp_buf, &(a->_mpfr_exp), sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_a += sizeof(mp_exp_t);
		memcpy(tmp_buf, a->_mpfr_d, (size_t)(sizeof(mp_limb_t) * _NUM_LIMB(a)));
		tmp_buf += sizeof(mp_limb_t) * _NUM_LIMB(a); ptr_a += sizeof(mp_limb_t *);
#else
		memcpy(tmp_buf, &(a->_mp_size), sizeof(int));
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
		memcpy(tmp_buf, &(a->_mp_prec), sizeof(int));
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
		memcpy(tmp_buf, &(a->_mp_exp), sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_a += sizeof(mp_exp_t);
		memcpy(tmp_buf, a->_mp_d, (size_t)(sizeof(mp_limb_t) * (a->_mp_prec + 1))); 
		tmp_buf += sizeof(mp_limb_t) * (a->_mp_prec + 1); ptr_a += sizeof(mp_limb_t *);
#endif
#else
#ifdef __MPFR_H
		MPI_Pack(&(a->_mpfr_size), 1, MPI_INT, buf, *bufsize, pos, comm);
		MPI_Pack(&(a->_mpfr_prec), 1, MPI_INT, buf, *bufsize, pos, comm);
		MPI_Pack(&(a->_mpfr_exp), 1, MPI_LONG, buf, *bufsize, pos, comm);
		MPI_Pack(a->_mpfr_d, _NUM_LIMB(a), MPI_UNSIGNED_LONG, buf, *bufsize, pos, comm);
#else
		MPI_Pack(&(a->_mp_size), 1, MPI_INT, buf, *bufsize, pos, comm);
		MPI_Pack(&(a->_mp_prec), 1, MPI_INT, buf, *bufsize, pos, comm);
		MPI_Pack(&(a->_mp_exp), 1, MPI_LONG, buf, *bufsize, pos, comm);
		MPI_Pack(a->_mp_d, a->_mp_prec + 1, MPI_UNSIGNED_LONG, buf, *bufsize, pos, comm);
#endif
#endif
		//(unsigned char *)a = (void *)ptr_a;
		//a = (mpf_ptr)ptr_a;
		a++;
	}

	//(mpf_ptr)a = tmp_a;
	a = tmp_a;
}

/* unpack mpf */
#ifdef __USE_MPI_PACK
void unpack_mpf(void *buf, int bufsize, int *pos, mpf_t ret, int count, MPI_Comm comm)
#else
void unpack_mpf(void *buf, mpf_t ret, int count)
#endif
{
	unsigned long prec;
	int i;
	//unsigned char *tmp_buf;
	unsigned char *tmp_buf;
	//mpf_ptr ptr_ret;
	unsigned char *ptr_ret;
	//mpf_ptr tmp_ret;
	mpf_ptr tmp_ret;

	//tmp_buf = (unsigned char *)buf;
	tmp_buf = (unsigned char *)buf;
	//ptr_ret = (mpf_ptr)ret;
	ptr_ret = (unsigned char *)ret;
	tmp_ret = (mpf_ptr)ret;

	for(i = 0; i < count; i++)
	{
#ifndef __USE_MPI_PACK
#ifdef __MPFR_H
		//mpf_set_prec(ret, mpfr_get_default_prec());
		memcpy(&(ret->_mpfr_prec), tmp_buf, sizeof(mpfr_prec_t));
		tmp_buf += sizeof(mpfr_prec_t); ptr_ret += sizeof(mpfr_prec_t);
#if MPFR_VERSION_MAJOR >= 2
#if MPFR_VERSION_MINOR >= 1
		memcpy(&(ret->_mpfr_sign), tmp_buf, sizeof(mpfr_sign_t));
		tmp_buf += sizeof(mpfr_sign_t); ptr_ret += sizeof(mpfr_sign_t);
#else
		memcpy(&(ret->_mpfr_size), tmp_buf, sizeof(int));
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
#endif
#else
		memcpy(&(ret->_mpfr_size), tmp_buf, sizeof(int));
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
#endif
		memcpy(&(ret->_mpfr_exp), tmp_buf, sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_ret += sizeof(mp_exp_t);

		//memcpy(ret->_mpfr_d, tmp_buf, (size_t)(sizeof(mp_limb_t) * ((mpfr_get_default_prec()-1)/GMP_LIMB_BITS + 1)));
		memcpy(ret->_mpfr_d, tmp_buf, (size_t)(sizeof(mp_limb_t) * _NUM_LIMB(ret)));
		tmp_buf += sizeof(mp_limb_t) * _NUM_LIMB(ret); ptr_ret += sizeof(mp_limb_t *);
#else
		memcpy(&(ret->_mp_size), tmp_buf, sizeof(int));
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
		memcpy(&(ret->_mp_prec), tmp_buf, sizeof(int));
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
		memcpy(&(ret->_mp_exp), tmp_buf, sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_ret += sizeof(mp_exp_t);
		memcpy(ret->_mp_d, tmp_buf, sizeof(mp_limb_t) * (ret->_mp_prec + 1));
		tmp_buf += sizeof(mp_limb_t) * (ret->_mp_prec + 1); ptr_ret += sizeof(mp_limb_t *);
#endif
#else
#ifdef __MPFR_H
		MPI_Unpack(buf, bufsize, pos, &ret->_mpfr_size, 1, MPI_INT, comm); 
		MPI_Unpack(buf, bufsize, pos, &ret->_mpfr_prec, 1, MPI_INT, comm); 
		MPI_Unpack(buf, bufsize, pos, &ret->_mpfr_exp, 1, MPI_LONG, comm); 
		MPI_Unpack(buf, bufsize, pos, ret->_mpfr_d, _NUM_LIMB(ret), MPI_UNSIGNED_LONG, comm);
#else
		MPI_Unpack(buf, bufsize, pos, &ret->_mp_size, 1, MPI_INT, comm); 
		MPI_Unpack(buf, bufsize, pos, &ret->_mp_prec, 1, MPI_INT, comm); 
		MPI_Unpack(buf, bufsize, pos, &ret->_mp_exp, 1, MPI_LONG, comm); 
		MPI_Unpack(buf, bufsize, pos, ret->_mp_d, ret->_mp_prec + 1, MPI_UNSIGNED_LONG, comm);
#endif
#endif
		//		(unsigned char *)ret = ptr_ret;
		//ret = (unsigned char *)ptr_ret;
		printf("unpack_mpf->"); mpf_out_str(stdout, 10, 0, ret); printf("\n");
		ret++;
	}
	//(mpf_ptr)ret = tmp_ret;
	ret = tmp_ret;
}

/* pack mpf and prec */
void pack_mpf2(mpf_t a, int incount, void *buf, unsigned long prec)
{
	int i;
	unsigned char *tmp_buf;
	unsigned char *ptr_a;
	mpf_ptr tmp_a;

	if(buf == NULL)
	{
		buf = allocbuf_mpf2(prec, incount);
	}

	tmp_buf = (unsigned char *)buf;
	ptr_a = (unsigned char *)a;
	tmp_a = (mpf_ptr)a;

	/* First, set "prec" */
	memcpy(tmp_buf, &prec, sizeof(unsigned long));
	tmp_buf += sizeof(unsigned long);

	for(i = 0; i < incount; i++)
	{
/* 2. int a->_mp_size; */
/* 3. int a->_mp_prec; */
/* 4. mp_exp_t(long?) a->_mp_exp; */
/* 5. mp_limb_t(unsigned long?) a->_mp_limb; */

#ifdef __MPFR_H
#if MPFR_VERSION_MAJOR >= 2
#if MPFR_VERSION_MINOR >= 1
		memcpy(tmp_buf, &(a->_mpfr_sign), sizeof(int));
#else
		memcpy(tmp_buf, &(a->_mpfr_size), sizeof(int));
#endif
#else
		memcpy(tmp_buf, &(a->_mpfr_size), sizeof(int));
#endif
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
		memcpy(tmp_buf, &(a->_mpfr_prec), sizeof(int));
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
		memcpy(tmp_buf, &(a->_mpfr_exp), sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_a += sizeof(mp_exp_t);
		memcpy(tmp_buf, a->_mpfr_d, (size_t)(sizeof(mp_limb_t) * _NUM_LIMB(a)));
		tmp_buf += sizeof(mp_limb_t) * _NUM_LIMB(a); ptr_a += sizeof(mp_limb_t *);
#else
		memcpy(tmp_buf, &(a->_mp_size), sizeof(int));
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
		memcpy(tmp_buf, &(a->_mp_prec), sizeof(int));
		tmp_buf += sizeof(int); ptr_a += sizeof(int);
		memcpy(tmp_buf, &(a->_mp_exp), sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_a += sizeof(mp_exp_t);
		memcpy(tmp_buf, a->_mp_d, (size_t)(sizeof(mp_limb_t) * (a->_mp_prec + 1))); 
		tmp_buf += sizeof(mp_limb_t) * (a->_mp_prec + 1); ptr_a += sizeof(mp_limb_t *);
#endif
//		(unsigned char *)a = ptr_a;
		a = (void *)ptr_a;
	}

	//(mpf_ptr)a = tmp_a;
	a = tmp_a;
}

/* unpack mpf and prec */
void unpack_mpf2(void *buf, mpf_t ret, int count, unsigned long *prec)
{
	int i;
	unsigned char *tmp_buf;
	unsigned char *ptr_ret;
	mpf_ptr tmp_ret;

	tmp_buf = (unsigned char *)buf;
	ptr_ret = (unsigned char *)ret;
	tmp_ret = (mpf_ptr)ret;

	/* First, get "prec" */
	memcpy(prec, tmp_buf, sizeof(unsigned long));
	tmp_buf += sizeof(unsigned long); ptr_ret += sizeof(unsigned long);

	for(i = 0; i < count; i++)
	{
#ifdef __MPFR_H
#if MPFR_VERSION_MAJOR >= 2
#if MPFR_VERSION_MINOR >= 1
		memcpy(&(ret->_mpfr_sign), tmp_buf, sizeof(int));
#else
		memcpy(&(ret->_mpfr_size), tmp_buf, sizeof(int));
#endif
#else
		memcpy(&(ret->_mpfr_size), tmp_buf, sizeof(int));
#endif
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
		memcpy(&(ret->_mpfr_prec), tmp_buf, sizeof(int));
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
		memcpy(&(ret->_mpfr_exp), tmp_buf, sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_ret += sizeof(mp_exp_t);
		memcpy(ret->_mpfr_d, tmp_buf, sizeof(mp_limb_t) * _NUM_LIMB(ret));
		tmp_buf += sizeof(mp_limb_t) * _NUM_LIMB(ret); ptr_ret += sizeof(mp_limb_t *);
#else
		memcpy(&(ret->_mp_size), tmp_buf, sizeof(int));
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
		memcpy(&(ret->_mp_prec), tmp_buf, sizeof(int));
		tmp_buf += sizeof(int); ptr_ret += sizeof(int);
		memcpy(&(ret->_mp_exp), tmp_buf, sizeof(mp_exp_t));
		tmp_buf += sizeof(mp_exp_t); ptr_ret += sizeof(mp_exp_t);
		memcpy(ret->_mp_d, tmp_buf, sizeof(mp_limb_t) * (ret->_mp_prec + 1));
		tmp_buf += sizeof(mp_limb_t) * (ret->_mp_prec + 1); ptr_ret += sizeof(mp_limb_t *);
		(unsigned char *)ret = ptr_ret;
#endif
	}
	//(mpf_ptr)ret = tmp_ret;
	ret = tmp_ret;
}



/* typedef and commit to mpich */
void commit_mpf(MPI_Datatype *mpi_mpf_t, unsigned long prec, MPI_Comm comm)
{
	int i, blockcounts[4];
	MPI_Datatype types[4];
	MPI_Aint displacements[4];

	mpf_t a;
	void *buf;
	int pos, bufsize;

	mpf_init2(a, prec);
	mpf_set_ui(a, 2);
	mpf_sqrt(a, a);

	/* struct of mpfr */
	/*   mpfr_prec_t(long) _mpfr_prec */
	/*   mpfr_sign_t(int)  _mpfr_sign */
	/*   mp_exp_t(long)  _mpfr_exp */
	/*   mp_limb_t(unsigned long) *_mpfr_d */

	blockcounts[0] = 1;
	types[0] = MPI_LONG;

	blockcounts[1] = 1;
	types[1] = MPI_INT;

	blockcounts[2] = 1;
	types[2] = MPI_LONG;

#ifdef __MPFR_H
	blockcounts[3] = _NUM_LIMB(a);
#else
	blockcounts[3] = a->_mp_prec + 1;
#endif
	types[3] = MPI_UNSIGNED_LONG;

	/* Pack */
	pos = 0;
	//buf = NULL;
	buf = allocbuf_mpf(prec, 1);
	pack_mpf(a, 1, buf);

	MPI_Address(buf, &displacements[0]);
	MPI_Address(buf + sizeof(long), &displacements[1]);
	MPI_Address(buf + sizeof(long) + sizeof(int), &displacements[2]);
	MPI_Address(buf + sizeof(long) + sizeof(int) + sizeof(mp_exp_t), &displacements[3]);
	displacements[3] -= displacements[0];
	displacements[2] -= displacements[0];
	displacements[1] -= displacements[0];
	displacements[0] = 0;

	/* print typemap */
//	for(i = 0; i < 4; i++)
//		printf("%d: (%d, %d) %d\n", i, types[i], blockcounts[i], displacements[i]);

	MPI_Type_struct(4, blockcounts, displacements, types, mpi_mpf_t);
	MPI_Type_commit(mpi_mpf_t);

	/* get size */
	MPI_Pack_size(1, *mpi_mpf_t, comm, &pos);
	//printf("size = %d\n", pos);

	mpf_clear(a);
	free(buf);
}

/* create op */
void create_mpf_op(MPI_Op *mpi_mpf_op, void (*func)(void *, void *, int *, MPI_Datatype *), MPI_Comm comm)
{
	/* operation create */
//	MPI_Op_create(func, MPI_COMM_WORLD, mpi_mpf_op);
	MPI_Op_create(func, comm, mpi_mpf_op);
#ifdef __MPFR_H
	mpfr_set_default_rounding_mode(GMP_RNDN);
#endif
}

/* clear type */
void free_mpf(MPI_Datatype *mpi_mpf_t)
{
	MPI_Type_free(mpi_mpf_t);
}

/* clear op */
void free_mpf_op(MPI_Op *mpi_mpf_op)
{
	MPI_Op_free(mpi_mpf_op);
}

/* Macro at each number of digits */
void use_mpi_mpf_b128(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B128), 128, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}

void use_mpi_mpf_b256(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B256), 256, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b512(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B512), 512, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b1024(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B1024), 1024, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b2048(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B2048), 2048, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b4096(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B4096), 4096, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b8192(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B8192), 8192, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b16384(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B16384), 16384, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b32768(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B32768), 32768, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b65536(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B65536), 65536, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_b131072(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_B131072), 131072, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}
void use_mpi_mpf_d50(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_D50), 167, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}

void use_mpi_mpf_d100(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_D100), 333, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}

void use_mpi_mpf_d1000(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_D1000), 3322, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}

void use_mpi_mpf_d10000(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_D10000), 33220, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}

void use_mpi_mpf_d100000(MPI_Comm comm)
{
	commit_mpf(&(MPI_MPF_D100000), 332193, comm);
	if(MPI_MPF_SUM == MPI_OP_NULL)
		create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, comm);
}

void free_mpi_mpf_b128(void)
{
	free_mpf(&(MPI_MPF_B128));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b256(void)
{
	free_mpf(&(MPI_MPF_B256));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b512(void)
{
	free_mpf(&(MPI_MPF_B512));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b1024(void)
{
	free_mpf(&(MPI_MPF_B1024));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b2048(void)
{
	free_mpf(&(MPI_MPF_B2048));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b4096(void)
{
	free_mpf(&(MPI_MPF_B4096));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b8192(void)
{
	free_mpf(&(MPI_MPF_B8192));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b16384(void)
{
	free_mpf(&(MPI_MPF_B16384));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b32768(void)
{
	free_mpf(&(MPI_MPF_B32768));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b65536(void)
{
	free_mpf(&(MPI_MPF_B65536));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_b131072(void)
{
	free_mpf(&(MPI_MPF_B131072));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_d50(void)
{
	free_mpf(&(MPI_MPF_D50));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_d100(void)
{
	free_mpf(&(MPI_MPF_D100));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_d1000(void)
{
	free_mpf(&(MPI_MPF_D1000));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_d10000(void)
{
	free_mpf(&(MPI_MPF_D10000));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}
void free_mpi_mpf_d100000(void)
{
	free_mpf(&(MPI_MPF_D100000));
	if(MPI_MPF_SUM != MPI_OP_NULL)
		free_mpf_op(&(MPI_MPF_SUM));
}

/* Operations for mpf_t */

/* mpf_add for MPI */
void _mpi_mpf_add(void *in, void *ret, int *len, MPI_Datatype *datatype)
{
	int i, itmp;
	unsigned long prec;
	void *ptr_in, *ptr_ret;
	mpf_t tmp_ret, tmp_in;

	
	ptr_in = (void *)in;
	memcpy(&itmp, ptr_in, sizeof(int));
	prec = _GET_PREC(itmp);

/*
	     if(*datatype == MPI_MPF_B128)	 prec = 128;
	else if(*datatype == MPI_MPF_B256)	 prec = 256;
	else if(*datatype == MPI_MPF_B512)	 prec = 512;
	else if(*datatype == MPI_MPF_B1024)	 prec = 1024;
	else if(*datatype == MPI_MPF_B2048)	 prec = 2048;
	else if(*datatype == MPI_MPF_B4096)	 prec = 4096;
	else if(*datatype == MPI_MPF_B8192)	 prec = 8192;
	else if(*datatype == MPI_MPF_B16384)	 prec = 16384;
	else if(*datatype == MPI_MPF_B32768)	 prec = 32768;
	else if(*datatype == MPI_MPF_B65536)	 prec = 65536;
	else if(*datatype == MPI_MPF_B131072)	 prec = 131072;
	else if(*datatype == MPI_MPF_D50)	 prec = MPF_D50;
	else if(*datatype == MPI_MPF_D100)	 prec = MPF_D100;
	else if(*datatype == MPI_MPF_D200)	 prec = MPF_D200;
	else if(*datatype == MPI_MPF_D400)	 prec = MPF_D400;
	else if(*datatype == MPI_MPF_D500)	 prec = MPF_D500;
	else if(*datatype == MPI_MPF_D1000)	 prec = MPF_D1000;
	else if(*datatype == MPI_MPF_D10000)	 prec = MPF_D10000;
	else if(*datatype == MPI_MPF_D100000)	 prec = MPF_D100000;
*/

	if(*datatype == MPI_MPF)
	{
		mpf_init(tmp_ret);
		mpf_init(tmp_in);
	}
	else
	{	
		mpf_init2(tmp_ret, prec);
		mpf_init2(tmp_in, prec);
	}

	ptr_in = (void *)in;
	ptr_ret = (void *)ret;

	for(i = 0; i < *len; i++)
	{
		/* unpack */
		unpack_mpf(ptr_ret, tmp_ret, 1);
		unpack_mpf(ptr_in, tmp_in, 1);

		/* add */
		mpf_add(tmp_ret, tmp_in, tmp_ret);

		/* pack */
		pack_mpf(tmp_ret, 1, (void *)ptr_ret);

		ptr_ret += get_bufsize_mpf(tmp_ret, 1);
		ptr_in += get_bufsize_mpf(tmp_in, 1);
	}

	mpf_clear(tmp_in);
	mpf_clear(tmp_ret);
}

/* mpf_add for mpf_t and prec*/
void _mpi_mpf2_add(void *in, void *ret, int *len, MPI_Datatype *datatype)
{
	int i;
	unsigned long prec;
	void *ptr_in, *ptr_ret;
	mpf_t tmp_ret, tmp_in;

	ptr_in = (void *)in;
	ptr_ret = (void *)ret;

	memcpy(&prec, ptr_in, sizeof(unsigned long));
	ptr_in += sizeof(unsigned long);
	memcpy(ptr_ret,  &prec, sizeof(unsigned long));
	ptr_ret += sizeof(unsigned long);

	mpf_init2(tmp_ret, prec);
	mpf_init2(tmp_in, prec);

	for(i = 0; i < *len; i++)
	{
		/* unpack */
		unpack_mpf(ptr_ret, tmp_ret, 1);
		unpack_mpf(ptr_in, tmp_in, 1);

		/* add */
		mpf_add(tmp_ret, tmp_in, tmp_ret);

		/* pack */
		pack_mpf(tmp_ret, 1, (void *)ptr_ret);

		ptr_ret += get_bufsize_mpf(tmp_ret, 1);
		ptr_in += get_bufsize_mpf(tmp_in, 1);
	}

	mpf_clear(tmp_in);
	mpf_clear(tmp_ret);
}
#endif
