# DATABASE ENDPOINT
QDRANTDB_URL = "http://103.176.178.107:6333"
DB_BATCH_SIZE = 256
DB_TIMEOUT = 60

# BM25
ENABLE_BM25 = False
BM25_TOP_K = 100

# EMBEDDING MODEL
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
EMBEDDING_DIM = 768
EMBEDDING_MAX_LENGTH = 256
EMBEDDING_TOP_K = 200

# LLM API ENDPOINT
LLM_MODEL = "ura-hcmut/ura-llama-7b"
TGI_URL = "http://localhost:10025"
API_KEY = "hf_sample_api_key"
MAX_ANSWER_LENGTH = 2048
MAX_MODEL_LENGTH = 4096
STOP_WORDS = ["</s>"]
REPETITION_PENALTY = 1.1

# FAQ HYPERPARAMETERS
# FAQ_FILE = "data/hcmut_data_faq.xlsx"
FAQ_FILE = "data/hcmut_tuyensinh_faq.csv"
FAQ_THRESHOLD = 80
FAQ_TEMPERATURE = 0.3
FAQ_TOP_P = 0.9
FAQ_TOP_K = 50
FAQ_PROMPT = """<s> [INST] <<SYS>> Hãy viết lại câu trả lời dùng thông tin bên dưới. <</SYS>>
Câu hỏi: {query}
Trả lời: {answers[0]["answer"]}

Câu trả lời được viết lại: [/INST]"""

# WEB HYPERPARAMETERS
WEB_FILE = "data/hcmut_data_web.json"
WEB_THRESHOLD = 50
WEB_TEMPERATURE = 0.6
WEB_TOP_P = 0.9
WEB_TOP_K = 50
WEB_PROMPT = """<s> [INST] <<SYS>> Trả lời câu hỏi bằng ngữ cảnh cho sẵn. <</SYS>>
Ngữ cảnh: '''
{join(documents, "\n")}
'''
Câu hỏi: {query}
Trả lời: [/INST]"""

# OTHERS
DEFAULT_ANSWERS = [
    "Xin lỗi! Tôi chưa có dữ liệu về câu hỏi bạn yêu cầu. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
    "Rất tiếc, tôi chưa có câu trả lời cho yêu cầu của bạn. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
    "Tôi chưa hiểu câu hỏi của bạn. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
]
