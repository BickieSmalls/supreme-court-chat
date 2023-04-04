from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
import openai
from gpt_index import download_loader
import os

from creds import open_ai_api_key
# set environment variable
os.environ["OPENAI_API_KEY"] = open_ai_api_key
openai.api_key = open_ai_api_key

PDFReader = download_loader("PDFReader")

loader = PDFReader()
documents = SimpleDirectoryReader('documents').load_data()
# documents = loader.load_data(file=Path('./documents/19-1392_6j37.pdf'))

# define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(client = openai,temperature=0, model_name="gpt-3.5-turbo"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=1000)

index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context
)

# save to disk
index.save_to_disk('index.json')

