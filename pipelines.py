# SETTING UP PIPELINE
#####################################################################################
import random
import logging
from envs import *
from haystack import Pipeline
from haystack.schema import Answer
from haystack.nodes import PromptModel, PromptNode, PromptTemplate
from haystack.nodes import (
    BM25Retriever,
    EmbeddingRetriever,
    SentenceTransformersRanker,
    Docs2Answers,
)
from invocation_layer import HFInferenceEndpointInvocationLayer
from custom_plugins import DocumentThreshold
from database import initialize_db

logger = logging.getLogger(__name__)


class ChatbotPipeline:
    def __init__(self, document_store):
        if ENABLE_BM25:
            retriever = BM25Retriever(document_store=document_store, top_k=BM25_TOP_K)

        embedding_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format="sentence_transformers",
            top_k=EMBEDDING_TOP_K,
        )

        document_store.update_embeddings(
            embedding_retriever, index="faq", batch_size=DB_BATCH_SIZE
        )
        document_store.update_embeddings(
            embedding_retriever, index="web", batch_size=DB_BATCH_SIZE
        )

        faq_threshold = DocumentThreshold(threshold=FAQ_THRESHOLD)
        web_threshold = DocumentThreshold(threshold=WEB_THRESHOLD)
        docs2answers = Docs2Answers()

        prompt_template_paraphrase = PromptTemplate(
            prompt=FAQ_PROMPT, output_parser={"type": "AnswerParser"}
        )

        prompt_template_ask = PromptTemplate(
            prompt=WEB_PROMPT, output_parser={"type": "AnswerParser"}
        )

        prompt_model = PromptModel(
            model_name_or_path=TGI_URL,
            api_key=API_KEY,
            max_length=MAX_ANSWER_LENGTH,
            invocation_layer_class=HFInferenceEndpointInvocationLayer,
            model_kwargs={
                "model_max_length": MAX_MODEL_LENGTH,
                "max_new_tokens": MAX_ANSWER_LENGTH,
                "repetition_penalty": REPETITION_PENALTY,
                "stream": True,
            },
        )

        prompt_paraphrase = PromptNode(
            model_name_or_path=prompt_model,
            default_prompt_template=prompt_template_paraphrase,
            api_key=API_KEY,
            max_length=MAX_ANSWER_LENGTH,
            top_k=FAQ_TOP_K,
            stop_words=STOP_WORDS,
            model_kwargs={
                "model_max_length": MAX_MODEL_LENGTH,
                "max_new_tokens": MAX_ANSWER_LENGTH,
                "temperature": FAQ_TEMPERATURE,
                "top_p": FAQ_TOP_P,
                "repetition_penalty": REPETITION_PENALTY,
                "stream": True,
            },
        )

        prompt_ask = PromptNode(
            model_name_or_path=prompt_model,
            default_prompt_template=prompt_template_ask,
            api_key=API_KEY,
            max_length=MAX_ANSWER_LENGTH,
            top_k=WEB_TOP_K,
            stop_words=STOP_WORDS,
            model_kwargs={
                "model_max_length": MAX_MODEL_LENGTH,
                "max_new_tokens": MAX_ANSWER_LENGTH,
                "temperature": WEB_TEMPERATURE,
                "top_p": WEB_TOP_P,
                "repetition_penalty": REPETITION_PENALTY,
                "stream": True,
            },
        )

        self.faq_pipeline = Pipeline()
        self.faq_params = {"EmbeddingRetriever": {"index": "faq"}}
        
        if ENABLE_BM25:
            self.faq_pipeline.add_node(
                component=retriever, name="Retriever", inputs=["Query"]
            )
            self.faq_params["Retriever"] = {"index": "faq"}

        self.faq_pipeline.add_node(
            component=embedding_retriever,
            name="EmbeddingRetriever",
            inputs=["Query" if not ENABLE_BM25 else "Retriever"],
        )
        self.faq_pipeline.add_node(
            component=faq_threshold, name="Threshold", inputs=["EmbeddingRetriever"]
        )
        self.faq_pipeline.add_node(
            component=docs2answers, name="Answer", inputs=["Threshold"]
        )
        self.faq_pipeline.add_node(
            component=prompt_paraphrase, name="prompt_node", inputs=["Answer.output_0"]
        )

        self.web_pipeline = Pipeline()
        self.web_params = {"EmbeddingRetriever": {"index": "web"}}
        if ENABLE_BM25:
            self.web_pipeline.add_node(
                component=retriever, name="Retriever", inputs=["Query"]
            )
            self.web_params["Retriever"] = {"index": "web"}
        self.web_pipeline.add_node(
            component=embedding_retriever,
            name="EmbeddingRetriever",
            inputs=["Query" if not ENABLE_BM25 else "Retriever"],
        )
        self.web_pipeline.add_node(
            component=web_threshold, name="Threshold", inputs=["EmbeddingRetriever"]
        )
        self.web_pipeline.add_node(
            component=prompt_ask, name="prompt_node", inputs=["Threshold"]
        )

    def __call__(self, query, **kwargs):
        return self.run(query, **kwargs)

    def run(self, query, **kwargs):
        if "params" not in kwargs:
            kwargs["params"] = {}

        kwargs["params"].update(self.faq_params)
        faq_ans = self.faq_pipeline.run(query, **kwargs)

        if len(faq_ans["answers"]) == 0 or faq_ans["answers"][0].answer.strip() == "":
            kwargs["params"].update(self.web_params)
            web_ans = self.web_pipeline.run(query, **kwargs)

            if (
                len(web_ans["answers"]) == 0
                or web_ans["answers"][0].answer.strip() == ""
            ):
                chosen_ans = random.choice(DEFAULT_ANSWERS)
                web_ans["answers"].append(Answer(chosen_ans, type="other"))

            return web_ans

        return faq_ans


def setup_pipelines(args):
    # Re-import the configuration variables
    from rest_api import config  # pylint: disable=reimported
    from rest_api.controller.utils import RequestLimiter

    pipelines = {}
    document_store = initialize_db(args)

    # Load query pipeline & document store
    print("[+] Setting up pipeline...")
    pipelines["query_pipeline"] = ChatbotPipeline(document_store)
    pipelines["document_store"] = document_store

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logger.info(
        "Concurrent requests per worker: %s", config.CONCURRENT_REQUEST_PER_WORKER
    )
    pipelines["concurrency_limiter"] = concurrency_limiter

    # Load indexing pipeline
    # index_pipeline, _ = _load_pipeline(config.PIPELINE_YAML_PATH, config.INDEXING_PIPELINE_NAME)
    # if not index_pipeline:
    #     logger.warning("Indexing Pipeline is not setup. File Upload API will not be available.")
    #     # Create directory for uploaded files
    #     os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)
    index_pipeline = None
    pipelines["indexing_pipeline"] = index_pipeline

    return pipelines
