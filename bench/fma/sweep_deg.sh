#!/bin/sh
# Degree sweep of the polynomial benchmark: how does the FMA/BF speedup behave
# for short polynomials?  Emits a CSV to bench/fma/results/poly_sweep.csv
cd "$(dirname "$0")/../.."   # repo root
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

OUT=bench/fma/results
CSV=$OUT/poly_sweep.csv
mkdir -p "$OUT"
BACKENDS=${BACKENDS:-"serial neon sve2"}
DEGS=${DEGS:-"2 3 4 6 8 12 16 24 32 64 128 256 512 1000"}

echo "backend,variant,deg,npts,rep,type,op,maxrelerr,meanrelerr,tpercall" > "$CSV"

for deg in $DEGS; do
	# keep the evaluated work per measurement roughly constant
	npts=$((4000000 / (deg + 1)))
	[ "$npts" -gt 400000 ] && npts=400000
	[ "$npts" -lt 2000 ] && npts=2000
	# POLYMUL is O(deg^2/2); keep rep*deg^2 roughly constant
	rep=$((20000000 / ((deg + 1) * (deg + 1))))
	[ "$rep" -gt 20000 ] && rep=20000
	[ "$rep" -lt 20 ] && rep=20
	echo "== deg=$deg npts=$npts rep=$rep =="
	for bk in $BACKENDS; do
		for v in "" "_bf" "_fma"; do
			case "$v" in
				"")     name=Q;;
				"_bf")  name=BF;;
				"_fma") name=FMA;;
			esac
			./bench/fma/poly_bench_${bk}${v} -deg "$deg" -npts "$npts" -rep "$rep" -accpts 200 -ntimed 5 2>/dev/null |
			awk -v bk="$bk" -v nm="$name" -v d="$deg" -v n="$npts" -v r="$rep" \
			    '!/^#/ && NF==6 && ($1=="dd"||$1=="td"||$1=="qd") \
			     { print bk","nm","d","n","r","$1","$2","$3","$4","$5 }' >> "$CSV"
		done
	done
done
echo "wrote $CSV"
