/********************************************************************************/
/*                                                                              */
/* test_urllink.c : get huge URI link matrix to check                           */
/*                                                                              */
/* Copyright (c) 2011 Tomonori Kouya, All rights reserved.                      */
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void usage(const char *progname)
{
	printf("Usage: %s #URIs SeedOfRnd\n", progname);
}

int main(int argc, char *argv[])
{
	long int num_uris, seed, total_num_data, i;

	if(argc < 3)
	{
		usage(argv[0]);
		return EXIT_SUCCESS;
	}

	num_uris = atol(argv[1]);
	seed = atol(argv[2]);

	srand((unsigned int)seed);

	total_num_data = num_uris * 100; // #URIs * Ave.#Links/1URI
	for(i = 0; i < total_num_data; i++)
		printf("%ld %ld\n", (long int)(rand() % num_uris), (long int)(rand() % num_uris));

	fprintf(stderr, "Num of URIs, Seed : %ld, %ld\n", num_uris, seed);
	fprintf(stderr, "Total Num of data : %ld\n", total_num_data);

	return EXIT_SUCCESS;
}
