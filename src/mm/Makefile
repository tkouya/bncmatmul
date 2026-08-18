#
# Matrix Market Matrix File I/O
#
include ../bncmake.inc

#all: libbncmm ex_mm_read ex_mm_write ex_mm_read_ex ex_mm_write_ex ex_readmatrix test_mm_cmat
all: libbncmm ex_mm_read ex_mm_write ex_mm_read_ex ex_mm_write_ex

.c.o: ../bnc.h ./bncsparse.h bncmm.h mmio.h
	$(CC) -c $(INCLUDES) $<

libbncmm: mmio.o readmatrix.o writematrix.o readwrite_clinear.o
	$(AR) libbncmm.a mmio.o
	$(AR) libbncmm.a readmatrix.o
	$(AR) libbncmm.a writematrix.o
	$(AR) libbncmm.a readwrite_clinear.o

ex_mm_read: example_read.o mmio.o
	$(CC) $(INCLUDES) example_read.o libbncmm.a $(LIBS) -o ex_mm_read

ex_mm_read_ex: example_read_ex.o mmio.o
	$(CC) $(INCLUDES) example_read_ex.o libbncmm.a $(LIBS) -o ex_mm_read_ex

ex_mm_write: example_write.o mmio.o
	$(CC) $(INCLUDES) example_write.o libbncmm.a $(LIBS) -o ex_mm_write

ex_mm_write_ex: example_write_ex.o mmio.o
	$(CC) $(INCLUDES) example_write_ex.o libbncmm.a $(LIBS) -o ex_mm_write_ex

ex_readmatrix: example_readmatrix.o mmio.o
	$(CC) $(INCLUDES) example_readmatrix.o libbncmm.a $(LIBSPARSE) $(LIBS) -o ex_readmatrix

test_mm_cmat: test_mm_cmat.o mmio.o
	$(CC) $(INCLUDES) test_mm_cmat.o libbncmm.a $(LIBSPARSE) $(LIBS) -o test_mm_cmat

clean:
	-rm *.o
	-rm *.a
	-rm ex_mm_read
	-rm ex_mm_write
	-rm ex_mm_read_ex
	-rm ex_mm_write_ex
	-rm ex_readmatrix
	-rm test_mm_cmat
