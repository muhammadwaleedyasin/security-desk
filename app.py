import streamlit as st
import openai
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import time

# Get the API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
# Function to load Files and change them into useable formats
def load_and_process_files(file_paths):
    documents = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
                text_data = df.to_string(index=False)
                text_file_path = file_path.replace('.csv', '.txt')
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(text_data)
                loader = TextLoader(text_file_path, encoding='utf-8')
            else:
                continue
            
            documents.extend(loader.load())
        except Exception as e:
            continue
    
    # Initialize the RecursiveCharacterTextSplitter with a chunk size of 1000 characters
    # and an overlap of 100 characters between consecutive chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split the list of documents into smaller text chunks based on the specified chunk size and overlap.
    # The result is stored in the 'texts' variable.
    texts = text_splitter.split_documents(documents)
    
    # Create vectorstore
    embeddings = OpenAIEmbeddings()
    # Creating a FAISS vector store from the text chunks using the generated embeddings
    # This allows efficient similarity search and retrieval of text chunks
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

# Example file paths. you can add as many as you want to
file_paths = ['intents-responses-01.csv','data.txt','output1.txt']
# Loading the files, process them, and create the vectorstore
vectorstore = load_and_process_files(file_paths)

# Creating a conversational chain, setting parameter according to our requirements
@st.cache_resource
def create_conversational_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        verbose=True,
        return_source_documents=False
    )

conversational_chain = create_conversational_chain()
# The title beeing displayed at the top of conversation page
st.title('Security Help Desk!')

# Initializing session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'untrained_response' not in st.session_state:
    st.session_state.untrained_response = None

# Displaying previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Get user input(This is where you will write your prompt)
prompt = st.chat_input('How can I assist you today?')

# Function to get responses from gpt 3.5 turbo
def get_gpt_response(prompt, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            *context,
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# Animation of appearing text like chatgpt interface
def display_response_animation(response, role):
    st.chat_message(role).markdown("")
    chat_placeholder = st.empty()
    words = response.split()
    display_text = ""
    for word in words:
        display_text += word + " "
        chat_placeholder.markdown(display_text)
        time.sleep(0.06)  # You can adjust this for faster/slower typing effect

if prompt:
    # Display the user's message in the chat interface and save it in the session state.
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Get the response from the conversational chain based on the user's prompt.
    response = conversational_chain({"question": prompt})
    
    # Check if the assistant's response indicates lack of information.
    assistant_response = response['answer']
    if "I'm sorry" in assistant_response or "I don't know" in assistant_response or "I don't have information" in assistant_response or "datasheet does not provide" in assistant_response or "does not provide specific information" in assistant_response or "text doesn't provide information" in assistant_response or "provided context does not include" in assistant_response or "document provided doesn't include" in assistant_response:
        # If response is unhelpful, use a ThreadPoolExecutor to get a GPT response.
        with ThreadPoolExecutor() as executor:
            gpt_future = executor.submit(get_gpt_response, prompt, st.session_state.messages)
            gpt_response = gpt_future.result()
        
        # Display the GPT response with animation and save it in the session state.
        display_response_animation(gpt_response, 'assistant')
        st.session_state.messages.append({'role': 'assistant', 'content': gpt_response})
    else:
        # Display the document-based response with animation and save it in the session state.
        display_response_animation(assistant_response, 'assistant')
        st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})

    # Get the untrained chatbot response and display it in the left sidebar.
    untrained_response = get_gpt_response(prompt, st.session_state.messages)
    st.session_state.untrained_response = untrained_response

# Displaying the untrained chatbot response in the left sidebar
with st.sidebar:
    st.header("Untrained Chatbot Response")
    if st.session_state.untrained_response:
        st.markdown(st.session_state.untrained_response)
