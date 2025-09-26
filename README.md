[Main python files]
- main_0926.py, fde_generator_optimized_stream.py

[Directory shapes]
- ./CACHE_DIR/DATASET_REPO_ID 에 'doc_embeds', 'queries' directory가 생길것임. 각 directory에 document/query 별 embedding 저장
- embedding 시간이 매우 오래걸리므로, 혹시나 다른 작업을 하다가 죽었을 때 embedding을 다시 해야하는 불상사를 막기 위함.
- _load_cache, _load_query_cache로 위 로직을 처리함. 당연히 처음 실행 때는 파일이 비어있을테니 그 떄 처음 embedding을 수행함

[FDE_generator 코드]
- 이것은 GPT의 힘을 빌렸고, 이유는 encode_document를 naive하게 하면 OOM이 발생함. FDE 작업 자체가 dimension이 커질수록, FDE memory footprint가 커져서 np.vstack 같이 한 번의 임베딩 오브젝트를 쓰는 작업은 OOM을 유발할 수 밖에 없음.
- np.vstack을 쓰지 않고 stream + mmap 방식으로 구현하도록 GPT에 요청하였음. 참고바람. 더 나은 로직을 짤 수 있다면 수정해도 무방함.
