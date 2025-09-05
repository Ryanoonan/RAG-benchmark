# RAG Benchmark for PopQA

A Retrieval-Augmented Generation (RAG) benchmark implementation using:
- **Retriever**: `intfloat/e5-base-v2` (dense retrieval)
- **Generator**: `Meta-Llama-3-8B-Instruct` (language model)
- **Dataset**: PopQA for question answering evaluation

### Quick start
```
pip install -r requirements.txt
conda install -c pytorch faiss-cpu=1.8.0
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

### Custom Configuration
```bash
python rag_benchmark.py --generator_model_path /path/to/model --index_path /path/to/faiss/index --passages /path/to/jsonl/corpus --num_samples num_samples --output_path my_results.json 
```
- **E5Retriever**: Dense retrieval using E5-base-v2 embeddings with FAISS indexing
- **TinyLlamaGenerator**: Text generation using Meta-Llama-3-8B-Instruct with RAG prompting


## Notes
- Use sample_corpus.jsonl and sample_index.faiss for testing.