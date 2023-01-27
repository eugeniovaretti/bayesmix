#!/usr/bin/env bash
build/run_mcmc \
    --algo-params-file examples/ar1nig_hierarchy/in/algo.asciipb \
    --hier-type AR1NIG --hier-args examples/ar1nig_hierarchy/in/ts.asciipb \
    --mix-type DPC2 --mix-args examples/ar1nig_hierarchy/in/dp_nig.asciipb \
    --coll-name examples/ar1nig_hierarchy/out/chains.recordio \
    --data-file resources/datasets/ts_emiliarom_last.csv \
    --mix-cov-file resources/datasets/coord_emiliarom.csv \
    --n-cl-file examples/ar1nig_hierarchy/out/numclust.csv \
    --clus-file examples/ar1nig_hierarchy/out/clustering.csv
