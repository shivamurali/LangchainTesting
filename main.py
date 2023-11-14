# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()
##################################################################################

#Select Dataset:
option = st.selectbox(
    'Select your Dataset',
    ('Hammurabi\'s Code', 'NDX Methodology', 'Goldman Sachs 2024 Outlook')
)
#Streamlit Page Creation:
st.title(option)

st.write('You Selected:',option)
# Create and load PDF Loader
if option == 'Hammurabi\'s Code':
    loader = PyPDFLoader('hammurabi_code.pdf')
elif option == 'NDX Methodology':
    loader = PyPDFLoader('NDX_metho.pdf')
elif option == 'Goldman Sachs 2024 Outlook':
    loader = PyPDFLoader('report.pdf')

# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name= "Random")

# Create vectorstore info object - metadata repo?clear

vectorstore_info = VectorStoreInfo(
    name="Goldman Sachs 2024 Outlook",
    description="Goldman Sachs 2024 Outlook as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    try:
        response = agent_executor.run(prompt)
    except:
        response = "There was an error with your query."
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 
    
