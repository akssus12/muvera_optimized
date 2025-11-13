#!/bin/bash

rm -rf ./cache_muvera/per_experiment_*
rm -rf ./cache_muvera/per_ndcg_*

for RC in 1000; do # 5 10 20 50 100
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  python3 baseline_naivecp_fullvector.py --p 4 --r 2 --nlist 1000 \
  -i "${FDE_PKL}" -o "${FAISS_OUT}" -rc "${RC}" -tk "${RC}" ${FORCE}
done