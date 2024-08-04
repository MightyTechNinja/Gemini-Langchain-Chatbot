# Gemini-Langchain-Chatbot
This repository consists of retrieval-based chatbot using Gemini API.

# Libraries Required
- LangChain
- Streamlit
- Faiss
# Data
I Extracted a small part of a story, it is attached in the repo. Feel free to give a different context while training.

# Procedure 
![image](https://github.com/user-attachments/assets/8cbebdf2-7dac-439b-bb4b-bb2fc026a23d)

## Backend Process 
1. **Text Input** : The text is extracted from a PDF by the PyPDF2.PdfReader library
2. **Text Chunking** : the text is divided into smaller segements called chunks by using the Recursive Character Text Splitter.
3. **Embedding Generation** : All the chunks are converted to embeddings uding the Google Generative AI Embedding API
4. **Storing Embeddings** : All the embeddings are converted into json format and stored using the FAISS indexing for easy retrieval.

## Frontend Process 
For the frontend part of the project Streamlit is used 
1. **Query Input** : Using the streamlit Input the user can give the query or ask the Question regarding the trained document.
2. **Query Embedding** : the query is converted into embeddings using the same method as earlier
3. **Similarity Search** : A search is run throught the stored embeddings to find the context related to the Query embeddings.
4. **API Respones Generation** : Based on the Query, Context and the Prompt the Api generates a response.
5. **Output Generation** : The Output given by the API is displayed on the frontend page.


