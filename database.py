# INITIALIZE DATABASE
#####################################################################################
import re
import pandas as pd
from envs import *
from tqdm import tqdm
from haystack.schema import Document
from haystack.nodes import PreProcessor
from haystack.document_stores import InMemoryDocumentStore
from qdrant_haystack import QdrantDocumentStore


def initialize_db(args):
    print("[+] Initialize database...")
    if args.dev:
        document_store = InMemoryDocumentStore(
            use_gpu=False, use_bm25=ENABLE_BM25, embedding_dim=EMBEDDING_DIM
        )
    else:
        document_store = QdrantDocumentStore(
            url=QDRANTDB_URL,
            embedding_dim=EMBEDDING_DIM,
            timeout=DB_TIMEOUT,
            embedding_field="embedding",
            hnsw_config={"m": 128, "ef_construct": 100},
            similarity="dot_product",
            recreate_index=(not args.no_reindex),
        )

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=EMBEDDING_MAX_LENGTH,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=int(EMBEDDING_MAX_LENGTH * 1.5),
    )

    if FAQ_FILE.endswith(".csv"):
        faq_df = pd.read_csv(FAQ_FILE)
    elif FAQ_FILE.endswith(".json"):
        faq_df = pd.read_json(FAQ_FILE)

    if not ("query" in faq_df.columns and "answer" in faq_df.columns):
        raise KeyError("FAQ file must have two keys 'query' and 'answer'")

    if WEB_FILE.endswith(".csv"):
        web_df = pd.read_csv(WEB_FILE)
    elif WEB_FILE.endswith(".json"):
        web_df = pd.read_json(WEB_FILE)

    if not ("text" in web_df.columns and "tables" in web_df.columns):
        raise KeyError("WEB file must have two keys 'text' and 'tables'")

    if args.dev:
        faq_df = faq_df.head(10)
        web_df = web_df.head(20)

    faq_documents = []
    idx = 0
    for _, d in tqdm(faq_df.iterrows(), desc="Loading FAQ..."):
        content = "Câu hỏi: " + d["query"] + " - Trả lời: " + d["answer"]
        faq_documents.append(Document(content=content, id=idx))
        idx += 1

    faq_documents = processor.process(faq_documents)
    document_store.write_documents(
        documents=faq_documents, index="faq", batch_size=DB_BATCH_SIZE
    )

    web_documents = []
    idx = 0
    for _, d in tqdm(web_df.iterrows(), desc="Loading web data..."):
        content = d["text"]
        web_documents.append(Document(content=content, id=idx))
        idx += 1

        if len(d["tables"]) > 0:
            for table in d["tables"]:
                web_documents.append(
                    Document(content=table, content_type="table", id=idx)
                )
                idx += 1

    web_documents = processor.process(web_documents)
    document_store.write_documents(
        documents=web_documents, index="web", batch_size=DB_BATCH_SIZE
    )

    return document_store
