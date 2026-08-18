#!/bin/bash

hostname=`hostname`
start_date=`date +%Y-%m-%d-%H-%S`

# data file name
data_fname="mpf_$hostname-$start_date.txt"
opt_fname="mpf_$hostname-$start_date-opt.csv"

echo "----- $data_fname -----" 2>&1 | tee -a $data_fname
echo "----- $opt_fname -----" 2>&1 | tee -a $data_fname

# num_threads
num_threads_array=(2 4 8)
#num_threads_array=(2 4)
#num_threads_array=(2)
#num_threads_array=() # no parallelization

# precision in bits
prec_array=(128 256 512 1024 2048)
#prec_array=(106 212)
#prec_array=(128 256 512 1024 2048 4096 16384 32768)
#prec_array=(106 128 212 256 333 512 1024 2048 3320 33320)

# dimension of matrices
#dim_array=(32 64 96)
dim_array=(128 256 512 1024)
#dim_array=(32 64 96 128 192 256 384 512 768 1024 1536 2048 3072)
#dim_array=(1536 2048 3072)

# maxsec
maxsec=600

# show your computational environment
echo " ---- Computational enviroment ----" 2>&1 | tee -a $data_fname
cat /proc/cpuinfo 2>&1 | tee -a $data_fname
cat /proc/meminfo 2>&1 | tee -a $data_fname
uname -a 2>&1 | tee -a $data_fname
gcc --version 2>&1 | tee -a $data_fname
g++ --version 2>&1 | tee -a $data_fname
icc --version 2>&1 | tee -a $data_fname
icpc --version 2>&1 | tee -a $data_fname
echo " ---- end of Computational enviroment ----" 2>&1 | tee -a $data_fname

echo "----- $data_fname -----" 2>&1 | tee -a $data_fname
echo "----- $opt_fname -----" 2>&1 | tee -a $data_fname

# main loop parallelised
echo " ---- mmbench start! ----" 2>&1 | tee -a $data_fname
for prec in ${prec_array[@]}; do
	for dim in ${dim_array[@]}; do
		in_dim_array=($((dim - 1)) $dim $((dim + 1)))
		for in_dim in ${in_dim_array[@]}; do
			echo "$prec bits, 1 thread, ($in_dim x $in_dim) * ($in_dim x $in_dim)" 2>&1 | tee -a $data_fname
			echo "./short_mmbench_mpf $prec $in_dim $in_dim $in_dim $maxsec $opt_fname"
			./short_mmbench_mpf $prec $in_dim $in_dim $in_dim $maxsec $opt_fname 2>&1 | tee -a $data_fname
			for num_threads in ${num_threads_array[@]}; do
				echo "$prec bits, $num_threads threads, ($in_dim x $in_dim) * ($in_dim x $in_dim)" 2>&1 | tee -a $data_fname
				echo "./short_mmbench_mpf_omp $prec $in_dim $in_dim $in_dim $num_threads $maxsec $opt_fname"
				./short_mmbench_mpf_omp $prec $in_dim $in_dim $in_dim $num_threads $maxsec $opt_fname 2>&1 | tee -a $data_fname
			done;
		done;
	done;
done;
echo " ---- mmbench done ! ----" 2>&1 | tee -a $data_fname

# ending

end_date=`date +%Y-%m-%d-%H-%S`
echo "$data_fname and $opt_fname has done!" 2>&1 | tee -a $data_fname
echo "start: $start_date" 2>&1 | tee -a $data_fname
echo "end  : $end_date" 2>&1 | tee -a $data_fname
echo "----- end of $data_fname -----" 2>&1 | tee -a $data_fname
