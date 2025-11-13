#!/bin/bash
# rm -rf ./cache_muvera/per_experiment_*
# --use_pack_half --half_policy stride2 --build_pack_if_missing --use_tpack --build_tpack_if_missing --tpack_chunk_cols 131072
# --use_pack_half(X): retriever._get_doc_embeddings(did, allow_build=True)
# --use_pack_half(O): retriever._get_doc_rows_half_from_pack(did, retriever.half_policy)
# ours_colgemm_fullvector_pread.py
# ours_colgemm_partialvector: partial vector(I/O) + columar layout(CP)
# ours_colgemm_es.py: Early stop(I/O) + columar layout(CP)
# ours_colgemm_sketch.py: sketch(I/O) + columar layout(CP)

# ours_colgemm_fullvector_npload.py
# ours_gemmcp_partialvector.py
# ours_gemmcp_fullvector.py 
export USE_IO_URING=1
export MAX_OVERREAD_PCT=0.10

for RC in 1000; do # 5 10 20 50 100 ours_colgemm_fullvector_iouring_panelcols_debugging
  for panel_cols in 256 512 1024 2048 4096 8192 16384 25600; do
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  python3 ours_colgemm_fullvector_iouring_panelcols_stream.py --p 4 --r 2 --nlist 1000 \
  -i "${FDE_PKL}" -o "${FAISS_OUT}" -rc "${RC}" -tk "${RC}" -rf 1 ${FORCE} --target_panel_cols "${panel_cols}" --iouring_submit_batch 128 --io_uring_qd 128 --use_pack_half --half_policy front --build_pack_if_missing --use_tpack --build_tpack_if_missing
  done
done