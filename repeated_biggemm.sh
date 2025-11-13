#!/bin/bash
rm -rf ./cache_muvera/per_experiment_*
# --use_pack_half --half_policy stride2 --build_pack_if_missing --use_tpack --build_tpack_if_missing --tpack_chunk_cols 131072
# --use_pack_half(X): retriever._get_doc_embeddings(did, allow_build=True)
# --use_pack_half(O): retriever._get_doc_rows_half_from_pack(did, retriever.half_policy)
# ours_colgemm_fullvector_pread.py:q

# ours_colgemm_fullvector_npload.py
# ours_biggemm_partialvector.py
# ours_biggemm_fullvector.py

for RC in 100; do # 5 10 20 50 100
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  python3 ours_biggemm_fullvector.py --p 4 --r 2 --nlist 1000 \
  -i "${FDE_PKL}" -o "${FAISS_OUT}" -rc "${RC}" -tk "${RC}" ${FORCE} --use_pack_half --half_policy front --build_pack_if_missing # --use_tpack --build_tpack_if_missing --tpack_chunk_cols 131072
done