#!/usr/bin/env bash

build/run_mcmc \
  --algo-params-file resources/tutorial/algo.asciipb \
  --hier-type LapNIG --hier-args resources/tutorial/lapnig_fixed.asciipb \
  --mix-type DP --mix-args resources/tutorial/dp_gamma.asciipb \
  --coll-name resources/tutorial/out/chains.recordio \
  --data-file resources/tutorial/data.csv \
  --grid-file resources/tutorial/grid.csv \
  --dens-file resources/tutorial/out/density_file.csv \
  --n-cl-file resources/tutorial/out/numclust.csv \
  --clus-file resources/tutorial/out/clustering.csv \
  #--best-clus-file resources/tutorial/out/best_clustering.csv
