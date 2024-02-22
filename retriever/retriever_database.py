# INITIALIZE DATABASE
#####################################################################################
import sys
import os
sys.path.append(os.getcwd())

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
            use_gpu=False, 
            use_bm25=ENABLE_BM25, 
            embedding_dim=EMBEDDING_DIM, 
            similarity="cosine" if args.cosine else "dot_product",
            index="faq"
        )
    else:
        document_store = QdrantDocumentStore(
            url=QDRANTDB_URL,
            embedding_dim=EMBEDDING_DIM,
            timeout=DB_TIMEOUT,
            embedding_field="embedding",
            hnsw_config={"m": 128, "ef_construct": 100},
            similarity="cosine" if args.cosine else "dot_product",
            recreate_index=(not args.no_reindex),
            index="faq"
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

    if FAQ_FILE.endswith(".xlsx"):
        faq_df = pd.read_excel(FAQ_FILE)
    if FAQ_FILE.endswith("csv"):
        faq_df = pd.read_csv(FAQ_FILE)
    else:
        raise TypeError("Input file 'FAQ_FILE' is not excel, please check again")

    # if args.dev:
    #     faq_df = faq_df.head(10)

    faq_documents = []
    idx = 0
    for _, d in tqdm(faq_df.iterrows(), desc="Loading FAQ..."):
        content = d["Question"]
        faq_documents.append(Document(content=content, id=idx, meta={'answer': d["Question"]}))
        idx += 1

    print(f"[+] FAQ_FILE rows: {faq_df.shape[0]} - FAQ_DOCUMENTS rows: {len(faq_documents)}")

    faq_documents = processor.process(faq_documents)
    document_store.write_documents(
        documents=faq_documents, index="faq", batch_size=DB_BATCH_SIZE
    )

    print(f"[+] FAQ_PROCESS: {len(faq_documents)} - DOCUMENT_STORE: {len(document_store.get_all_documents())}")

    return document_store