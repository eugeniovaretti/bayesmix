#!/usr/bin/env bash
for M in 0.01 0.1 1 3 10
do
  for A in 0.18 0.53 1.02 1.15 1.66 2.57 3.33 5
  do
    touch examples/ar1nig_hierarchy/inlist/dp_M_"$M"_A_"$A".asciipb
    printf "fixed_value {\n  totalmass: "$M"\n  a: "$A"\n}" > examples/ar1nig_hierarchy/inlist/dp_M_"$M"_A_"$A".asciipb
    build/run_mcmc \
    --algo-params-file examples/ar1nig_hierarchy/in/algo.asciipb \
    --hier-type AR1NIG --hier-args examples/ar1nig_hierarchy/in/ts.asciipb \
    --mix-type DPC2 --mix-args examples/ar1nig_hierarchy/inlist/dp_M_"$M"_A_"$A".asciipb \
    --coll-name examples/ar1nig_hierarchy/outlist/chains.recordio \
    --data-file resources/datasets/ts_emiliarom_last.csv \
    --mix-cov-file resources/datasets/coord_emiliarom.csv \
    --n-cl-file examples/ar1nig_hierarchy/outlist/numclust_M_"$M"_A_"$A".csv \
    --clus-file examples/ar1nig_hierarchy/outlist/clustering_M_"$M"_A_"$A".csv
  done
done
