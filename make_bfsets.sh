#!/bin/bash

DT="treccovid arguana scidocs fiqa"
for i in $DT; do
  for RC in 250 500 750 1000; do # "${RC}"
    python3 indexing_fdeivf_search_basedbf.py --p 4 --r 2 --nlist 1000 \
  -i "${FDE_PKL}" -o "${FAISS_OUT}" -rc "${RC}" -tk "${RC}" -dt "${i}"  ${FORCE}
  done
done
