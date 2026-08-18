// test_qd.cc : test program for QD library
#include <iostream>
#include <iomanip>
#include "qd/qd_real.h" // QD

int main(void)
{
    qd_real a = 2.0;
    qd_real r = (1.0 / std::sqrt(a.x[0]));
    std::cout << std::setprecision(32) << "1/a.x[0] = " << r << std::endl;
    std::cout << std::setprecision(32) << "double   = " << (double)1.0 / std::sqrt(2.0)
 << std::endl;
    return 0;
}
