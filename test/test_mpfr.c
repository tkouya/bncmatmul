include <stdio.h>
#include "mpfr.h"

int main(void)
{
        mpfr_t a, b, c;

        mpfr_init2(a, 128);
        mpfr_init2(b, 256);
        mpfr_init2(c, 512);

        mpfr_sqrt_ui(a, 3UL, MPFR_RNDN);
        mpfr_sqrt_ui(b, 2UL, MPFR_RNDN);

        mpfr_add(c, a, b, MPFR_RNDN);
        mpfr_printf("a = %RNe\n", a);
        mpfr_printf("b = %RNe\n", b);
        mpfr_printf("c = %RNe\n", c);

        mpfr_clear(a);
        mpfr_clear(b);
        mpfr_clear(c);

        return 0;
}
