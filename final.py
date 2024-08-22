import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from bs4 import BeautifulSoup
from langchain.schema import Document
import validators
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re
import os




# Custom CSS for better styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .css-1v3fvcr { 
        background-color: #f0f2f6;
    }
    .stTextInput div[data-baseweb="input"] {
        border-radius: 8px;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    .stFileUploader>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Page title with styling
st.title("Document & Web Summarizer")
st.markdown("Welcome to the Document & Web Summarizer! Enter your YouTube video URLs, website URLs, or upload PDF files to get concise summaries.")

# Sidebar for Groq API Key and Search Query


st.subheader("Search Query")
search_query = st.text_input("Enter the topic or keyword to search for", placeholder="e.g., machine learning")

# Gemma Model Using Groq API
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt Template for Summarization
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Layout for inputs using columns
st.header("Input Sections")



st.subheader("YouTube Video URLs")
video_urls = st.text_area("Enter YouTube Video URLs (one per line)", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ", height=200)

st.subheader("Website URLs")
website_urls = st.text_area("Enter Website URLs (one per line)", placeholder="e.g., https://www.example.com", height=200)

st.subheader("Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)


# Helper function to fetch and parse website content
def fetch_website_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        st.error(f"An error occurred while fetching the website content: {e}")
        return None

# Function to filter content based on search query
def filter_content(content, query):
    if query:
        # Case insensitive search for query in content
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        filtered = "\n".join([line for line in content.split('\n') if pattern.search(line)])
        return filtered
    return content

# Helper function to clean text
def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

# Function to summarize text with Groq API
def summarize_text(text):
    prompt_text = prompt.format(text=text)
    try:
        # Send prompt to the model
        response = llm.generate(messages=[{"role": "user", "content": prompt_text}])
        return response['choices'][0]['text']  # Adjust as necessary based on the actual API response
    except Exception as e:
        st.error(f"An error occurred while summarizing text: {e}")
        return "Error during summarization."
    
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text        


# Button to trigger summarization
if st.button("Summarize Content", key="summarize"):
    # Validate Groq API Key
    if not groq_api_key.strip():
        st.error("Please provide the API key.")
        st.stop()

    combined_documents = []

    # Summarize YouTube Videos
    if video_urls.strip():
        video_urls_list = [url.strip() for url in video_urls.split("\n") if validators.url(url) and "youtube.com" in url]
        for url in video_urls_list:
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
                filtered_content = filter_content(content, search_query)
                cleaned_docs = [Document(page_content=clean_text(doc.page_content)) for doc in docs]
                combined_documents.extend(cleaned_docs)
                if filtered_content:
                    summary = summarize_text(filtered_content)
                    st.success(f"Summary of {url}:")
                    st.write(filtered_content)
                else:
                    st.warning(f"No relevant content found in {url} for the query: {search_query}")
            except Exception as e:
                st.error(f"An error occurred with {url}: {e}")


                
    # Summarize Websites
    if website_urls.strip():
        website_urls_list = [url.strip() for url in website_urls.split("\n") if validators.url(url)]
        for url in website_urls_list:
            content = fetch_website_content(url)
            if content:
                filtered_content = filter_content(content, search_query)
                cleaned_content = clean_text(content)
                combined_documents.append(Document(page_content=cleaned_content))
                if filtered_content:
                    doc = Document(page_content=filtered_content)
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run([doc])
                    st.success(f"Summary of {url}:")
                    st.write(summary)
                else:
                    st.warning(f"No relevant content found on {url} for the query: {search_query}")
        
    if uploaded_files: 
      for uploaded_file in uploaded_files:
       raw_text=get_pdf_text([uploaded_file])
      filtered_content = filter_content(raw_text,search_query)
      if filtered_content:
            st.success(f"Summary of pdf:")
            st.write(filtered_content)
      else:
            st.warning(f"No relevant content found for the query: {search_query}")
    


