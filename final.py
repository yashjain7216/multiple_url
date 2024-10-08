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
    @keyframes slide-in {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    .slide-in-text {
        animation: slide-in 3s ease-out;
        font-size: 24px;
        color: white;
        font-weight: bold;
        margin-bottom: 20px;
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
st.title("Summarizer")
st.markdown('<div class="slide-in-text">Hi, how can I help you to summarize data?</div>', unsafe_allow_html=True)

# Sidebar for Groq API Key and Search Query


st.subheader("Topic or Title")
search_query = st.text_input("Enter the topic or keyword to search for", placeholder="Type here")

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


content_type = st.selectbox("Select Content Type to Summarize", ("YouTube Videos", "Websites", "PDF Files"))

if content_type == "YouTube Videos":
    st.subheader("YouTube Video URLs")
    video_urls = st.text_area("Enter YouTube Video URLs (one per line)", placeholder="Enter urls", height=200)

elif content_type == "Websites":
    st.subheader("Website URLs")
    website_urls = st.text_area("Enter Website URLs (one per line)", placeholder="Enter websites url", height=200)


# File uploaders if PDFs or Text Files are selected
elif content_type == "PDF Files":
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
# elif content_type == "Text Files":
#     text_files = st.file_uploader("Upload Text files", type=["txt"], accept_multiple_files=True)



# Step 2: Define the summarization functionality when the button is clicked
if st.button("Summarize Content", key="summarize"):
    # Validate Groq API Key
    if not groq_api_key.strip():
        st.error("Please provide the API key.")
        st.stop()

    combined_documents = []

    if content_type == "YouTube Videos":
        
        # Summarize YouTube Videos
        if video_urls.strip():
            video_urls_list = [url.strip() for url in video_urls.split("\n") if validators.url(url) and "youtube.com" in url]
            for url in video_urls_list:
                try:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    docs = loader.load()
                    content = "\n".join([doc.page_content for doc in docs])
                    filtered_content = filter_content(content, search_query)
                    if filtered_content:
                        doc = Document(filtered_content)
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        summary = chain.run([doc])
                        st.success(f"Summary of {url}:")
                        st.text_area("Summary", summary, height=200, disabled=True)
                    else:
                        st.warning(f"No relevant content found in {url} for the query: {search_query}")
                except Exception as e:
                    st.error(f"An error occurred with {url}: {e}")

    elif content_type == "Websites":
        
        # Summarize Websites
        if website_urls.strip():
            website_urls_list = [url.strip() for url in website_urls.split("\n") if validators.url(url)]
            for url in website_urls_list:
                try:
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
                            st.text_area("Summary", summary, height=200, disabled=True)
                        else:
                            st.warning(f"No relevant content found on {url} for the query: {search_query}")
                except Exception as e:
                    st.error(f"An error occurred with {url}: {e}")

    elif content_type == "PDF Files":
        
        # Summarize PDF Files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    raw_text = get_pdf_text([uploaded_file])
                    filtered_content = filter_content(raw_text, search_query)
                    if filtered_content:
                        st.success(f"Summary of PDF:")
                        st.text_area("Summary", filtered_content, height=200, disabled=True)
                    else:
                        st.warning(f"No relevant content found for the query: {search_query}")
                except Exception as e:
                    st.error(f"An error occurred with the PDF: {e}")
