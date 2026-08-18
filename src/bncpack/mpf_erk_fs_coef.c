/*                                                 */
/* Explicit Runge-Kutta Method with fixed stepsize */
/*                                                 */
void mpf_erk_fs_coef(FILE *fp, mpf_t x, MPFVector y, mpf_t x0, MPFVector y0, long int div_num, void(* func)(MPFVector y, mpf_t x0, MPFVector y0), MPFVector coef_c, MPFMatrix coef_a, MPFVector coef_w)
{
	long int steps, i;
	MPFVector old_y, tmp_y, lf_tmp, k[7];
	mpf_t tmp[3], new_x, old_x, in_h;


//	printf("c: \n"); print_mpfvector(coef_c);
//	printf("w: \n"); print_mpfvector(coef_w);
//	printf("a: \n"); print_mpfmatrix(coef_a);

	/* init */
	for(i = 0; i < coef_w->dim; i++)
		k[i] = init2_mpfvector(y->dim, y->prec);
	old_y = init2_mpfvector(y->dim, y->prec);
	tmp_y = init2_mpfvector(y->dim, y->prec);
	lf_tmp = init2_mpfvector(y->dim, y->prec);
	mpf_init2(new_x, mpf_get_prec(x));
	mpf_init2(old_x, mpf_get_prec(x));
	mpf_init2(tmp[0], mpf_get_prec(x));
	mpf_init2(tmp[1], mpf_get_prec(x));
	mpf_init2(tmp[2], mpf_get_prec(x));
	mpf_init2(in_h, mpf_get_prec(x));

	/* set */
	mpf_set(old_x, x0);
	subst_mpfvector(old_y, y0);

	/* check interval */
	/* in_max_h := min((x - x0) / 2, max_h) */
	mpf_sub(in_h, x, x0);
	if(div_num <= 0)
	{
		fprintf(stderr, "mpf_erk_fs: Number of division is illegal.\n");
		return;
	}
	mpf_div_ui(in_h, in_h, (unsigned long)div_num);

	/* output */
	if(fp != NULL)
	{
		printf("           x             ");
		for(i = 0; i < y->dim; i++)
			printf("         y[%5d]          ", i);
		printf("\n");
	}

	/* main loop */
	for(steps = 0; steps < div_num; steps++)
	{

		/* calc */

		/* tmp_y := y0 + h * erk(x0, y0) */
		mpf_erk_1step(tmp_y, old_x, old_y, in_h, func, coef_c, coef_a, coef_w, lf_tmp, k);

		/* set new x, y */
		mpf_add(old_x, old_x, in_h);
		subst_mpfvector(old_y, tmp_y);

		/* output */
		if(fp != NULL)
		{
			printf("%25.17e ", mpf2double(old_x));
			for(i = 0; i < old_y->dim; i++)
				printf("%25.17e ", mpf2double(gmpfvi(old_y, i)));
			printf("\n");
		}

	}

	/* finish! */
	subst_mpfvector(y, old_y);

	printf("OK! mpf_erk_fs has been finished.\n");
	printf("Stepsize  : ");
		 mpf_out_str(stdout, 10, 0, in_h);
		 printf("\n");
	printf("Number of steps  : %ld\n", steps);
	printf("Integral interval : ");
		 printf("[ ");
		 mpf_out_str(stdout, 10, 0, x0);
		 printf(", ");
		 mpf_out_str(stdout, 10, 0, old_x);
		 printf(" ]\n");
	printf("Numerical solution: \n");
		 print_mpfvector(y);
		 printf("\n");
//	mpf_set(x, old_x);

	/* clear */
	free_mpfvector(old_y);
	free_mpfvector(tmp_y);
	free_mpfvector(lf_tmp);
	mpf_clear(new_x);
	mpf_clear(old_x);
	mpf_clear(tmp[0]);
	mpf_clear(tmp[1]);
	mpf_clear(tmp[2]);
	mpf_clear(in_h);
	for(i = 0; i < coef_w->dim; i++)
		free_mpfvector(k[i]);

}
