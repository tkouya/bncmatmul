/********************************************************************************/
/* fread_write_complex.c: Read and Write files to store *array, *poly.          */
/*                                                                              */
/* Copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
//#include "bnc.h"
#include "bncmatmul.h"

#ifdef __cpluplus
extern "C" {
#endif // __cplusplus

#ifndef BNC_IO_MAX_DEC_DIGITS
#define BNC_IO_MAX_DEC_DIGITS 65536
#endif // BNC_IO_MAX_DEC_DIGITS

// read complex numbers from fp
void fread_cmpfarray(FILE *fp, CMPFArray array)
{
	long int i, index;
	char line_buf[BNC_IO_MAX_DEC_DIGITS];
	char tmpsrc_real[BNC_IO_MAX_DEC_DIGITS]; /* maximum digits */
   	char tmpsrc_imag[BNC_IO_MAX_DEC_DIGITS]; /* maximum digits */
	//MPFCmplx tmp;
    mpc_t tmp;
    mpf_t rtmp;

	//mpf_init2(tmp, array->prec);
    mpf_init2(rtmp, array->prec);
    //tmp = init2_mpfcmplx(array->prec);
    mpc_init2(tmp, array->prec);

	for(i = 0; i < array->size; i++)
	{
		//fscanf(fp, "%ld, %s, %s", &index, tmpsrc_real, tmpsrc_imag); // obsolete
		// new!
		if(fgets(line_buf, sizeof(line_buf), fp) != NULL)
		{
			// https://kaworu.jpn.org/kaworu/2007-12-27-2.php
			sscanf(line_buf, "%ld, %[^,], %[^,]", &index, tmpsrc_real, tmpsrc_imag);
			mpf_set_str(rtmp, tmpsrc_real, 10); mpf_set(mpc_realref(tmp), rtmp); //set_real_mpfcmplx(tmp, rtmp);
			mpf_set_str(rtmp, tmpsrc_imag, 10); mpf_set(mpc_imagref(tmp), rtmp); //set_image_mpfcmplx(tmp, rtmp);
			set_cmpfarray_i(array, i, tmp);
		}
	}

	//mpf_clear(tmp);
    mpf_clear(rtmp);
    //free_mpfcmplx(tmp);
    mpc_clear(tmp);
}

void fread_cmpfarray_fname(const char *fname, CMPFArray array)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_cmpfarray(fp, array);

	fclose(fp);
}

void fwrite_cmpfarray(FILE *fp, CMPFArray array)
{
	long int i;
    mpf_t rtmp;
    //MPFCmplx tmp;
    mpc_t tmp;

    mpf_init2(rtmp, array->prec);
    //tmp = init2_mpfcmplx(array->prec);
    mpc_init2(tmp, array->prec);

	for(i = 0; i < array->size; i++)
	{
		fprintf(fp, "%ld, ", i);
        //get_real_mpfcmplx(rtmp, get_cmpfarray_i(array, i));
		mpf_out_str(fp, 10, 0, mpc_realref(get_cmpfarray_i(array, i))); // rtmp);
        fprintf(fp, ", ");
        //get_image_mpfcmplx(rtmp, get_cmpfarray_i(array, i));
		mpf_out_str(fp, 10, 0, mpc_imagref(get_cmpfarray_i(array, i))); // rtmp);
		fprintf(fp, "\n");
	}

    mpf_clear(rtmp);
    //free_mpfcmplx(tmp);
    mpc_clear(tmp);
}

void fwrite_cmpfarray_fname(const char *fname, CMPFArray array)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fwrite_cmpfarray(fp, array);

	fclose(fp);
}

void fread_mpfpolycoef(FILE *fp, MPFPoly p, long int maxdeg)
{
	long int i, index;
	char tmpsrc[BNC_IO_MAX_DEC_DIGITS]; /* maximum degits */
	mpf_t tmp;
	int ret_fscanf; // 2025-07-09(Wed) T.Kouya


	mpf_init2(tmp, p->prec);

	for(i = 0; i <= maxdeg; i++)
	{
		//fscanf(fp, "%ld, %s", &index, tmpsrc);
		ret_fscanf = fscanf(fp, "%ld %s", &index, tmpsrc);
		if(ret_fscanf == 2)
		{
	//		printf("%d: %s\n", index, tmpsrc);
			mpf_set_str(tmp, tmpsrc, 10);
			set_mpfpoly_i(p, index, tmp);
		}
	}

	mpf_clear(tmp);
}

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
