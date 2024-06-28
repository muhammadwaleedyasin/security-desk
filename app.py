import streamlit as st
import openai
import pandas as pd
import os
import time
import pickle
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
from openai import RateLimitError
from langchain_openai import OpenAIEmbeddings
# Load .env variable from local directory
load_dotenv()

# Get the API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def process_in_batches(texts, batch_size=50):
    all_embeddings = []
    all_texts = []
    progress_bar = st.progress(0)
    
    embeddings = OpenAIEmbeddings()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        retry_attempts = 5
        retry_delay = 1  # Initial delay in seconds
        
        for attempt in range(retry_attempts):
            try:
                batch_embeddings = embeddings.embed_documents([t.page_content for t in batch])
                all_embeddings.extend(batch_embeddings)
                all_texts.extend([t.page_content for t in batch])
                break
            except RateLimitError as e:
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e
        
        # Update progress bar
        progress = (i + batch_size) / len(texts)
        progress_bar.progress(min(progress, 1.0))
        
        # Add a small delay between batches
        time.sleep(5)
    
    # Create a single FAISS index from all embeddings
    vectorstore = FAISS.from_texts(all_texts, embeddings, metadatas=[{"source": i} for i in range(len(all_texts))])
    
    progress_bar.empty()
    
    return vectorstore

def load_and_process_files(file_paths):
    documents = []
    processed_files_count = 0
    progress_placeholder = st.empty()
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
            processed_files_count += 1
            
            # Update the counter in the sidebar
            progress_placeholder.markdown(f"**Number of processed files:** {processed_files_count}")
        except Exception as e:
            continue
    
    # Initialize the RecursiveCharacterTextSplitter with a chunk size of 1000 characters
    # and an overlap of 100 characters between consecutive chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split the list of documents into smaller text chunks based on the specified chunk size and overlap.
    texts = text_splitter.split_documents(documents)
    
    # Process texts in batches
    vectorstore = process_in_batches(texts)
    
    # Remove the progress counter once all files are processed
    progress_placeholder.empty()
    
    return vectorstore

def save_vectorstore(vectorstore, file_path='vectorstore.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(file_path='vectorstore.pkl'):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_or_create_vectorstore(file_paths, vectorstore_path='vectorstore.pkl'):
    if os.path.exists(vectorstore_path):
        st.info("Loading existing vectorstore...")
        return load_vectorstore(vectorstore_path)
    else:
        st.info("Creating new vectorstore...")
        vectorstore = load_and_process_files(file_paths)
        save_vectorstore(vectorstore, vectorstore_path)
        return vectorstore

# Example file paths. Add your file paths here
file_paths = ['intents-responses-01.csv','data.txt','output.txt']
# Loading the files, process them, and create the vectorstore
vectorstore = load_or_create_vectorstore(file_paths)

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

# The title being displayed at the top of conversation page
st.title('Security Help Desk!')

# Initializing session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'untrained_response' not in st.session_state:
    st.session_state.untrained_response = None

# Displaying previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Get user input (This is where you will write your prompt)
prompt = st.chat_input('How can I assist you today?')

# Function to get responses from GPT-3.5 Turbo
def get_gpt_response(prompt, context):
    retry_attempts = 5
    retry_delay = 1  # Initial delay in seconds

    for attempt in range(retry_attempts):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *context,
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message['content']
        except RateLimitError as e:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise e

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
    assistant_response = response.get('answer', "I'm sorry, I don't have information regarding that.")
    if "I'm sorry" in assistant_response or "I don't have information" in assistant_response or "datasheet does not provide" in assistant_response or "does not provide specific information" in assistant_response or "text doesn't provide information" in assistant_response or "provided context does not include" in assistant_response or "document provided doesn't include" in assistant_response:
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
