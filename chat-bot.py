import streamlit as st
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
import openai
import os

# user input history
if "user_input_hist" not in st.session_state:
    st.session_state.user_input_hist = []
if "chatbot_response_hist" not in st.session_state:
    st.session_state.chatbot_response_hist = []

# get openai api key from environment variable
#open_ai_api_key = os.environ.get("OPEN_AI_API_KEY")
from creds import open_ai_api_key
openai.api_key = open_ai_api_key
os.environ["OPENAI_API_KEY"] = open_ai_api_key

index = GPTSimpleVectorIndex.load_from_disk('index.json')

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

# user input
user_input = st.text_input("Enter your message")

# chatbot response if button is pressed
if user_input:
    st.session_state.user_input_hist.append(user_input)
    # chatbot response
    chatbot_response = index.query(user_input, mode="embedding", service_context=service_context, similarity_top_k=3)
    st.session_state.chatbot_response_hist.append(chatbot_response)

    st.write(chatbot_response)
