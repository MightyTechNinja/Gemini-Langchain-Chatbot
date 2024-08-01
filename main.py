import numpy as np
import pandas as pd

import google.generativeai as genai
import json
import os
import faiss
import ai21
import streamlit as st
from typing import List,Tuple
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


API_KEY = 'AIzaSyAaBfBlz2WJx2oz9d2s11ufTubrjJADB0M'
genai.configure(api_key=os.getenv("API_KEY"))


pdf_file = open('test2.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)
num_pages = len(pdf_reader.pages)
text = ''
for page_num in range(num_pages):
    page = pdf_reader.pages[page_num]
    text += page.extract_text()

pdf_file.close()

print(text)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_text(text)
print(chunks[0])
len(chunks)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
embeddings = embedding_model.embed_documents(chunks)
embeddings[0]


def save_chunks_and_embeddings_to_json(chunks, embeddings, output_file):
    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            'chunk': chunk,
            'embedding': embeddings[i]
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


save_chunks_and_embeddings_to_json(chunks,embeddings,'chunksandembeddings.json')

with open('chunksandembeddings.json','r',encoding='utf-8') as f:
    data = json.load(f)
    embeddings1 = [item['embedding']for item in data]

embeddings_np = np.array(embeddings1, dtype=np.float32)
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)
faiss.write_index(index, 'faiss_index.index')


index = faiss.read_index("faiss_index.index")

with open("chunksandembeddings.json", "r",encoding="utf-8") as f:
    data = json.load(f)
    chunks1 = [item['chunk'] for item in data]
    
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY)
prompt_template = """You are a helpful assistant that answers questions based on the provided context. and you can be creative if necessary
Context: {context}
Question: {question}
Answer: """

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain = LLMChain(llm=llm, prompt=prompt)


def app():
    st.title("PDF Chatbot")
    query = st.text_input("Enter your query")
    if query:
        query_embeddings = embedding_model.embed_query([query])
        query_embedding_array = np.array(query_embeddings).astype('float32')
        if len(query_embedding_array.shape) == 1:
            query_embedding_array = query_embedding_array.reshape(1, -1)
        search_result = index.search(query_embedding_array, k=5)
        distances = search_result[0]
        indices = search_result[1]
        relevant_indices = indices[0]
        relevant_embeddings = [embeddings1[idx] for idx in relevant_indices]

        flattened_relevant_embeddings = [np.reshape(emb, -1) for emb in relevant_embeddings]
        similarity_scores = cosine_similarity([query_embedding_array.flatten()], flattened_relevant_embeddings)[0]

        sorted_indices = np.argsort(-similarity_scores)
        sorted_embeddings = [relevant_embeddings[idx] for idx in sorted_indices]
        sorted_texts = [chunks1[relevant_indices[idx]] for idx in sorted_indices]

        context = " ".join(sorted_texts)

        # similarity_scores = cosine_similarity(query_embedding_array, chunks1)
        # relevant_chunk_indices = np.argsort(similarity_scores[0])[-5:][::-1]
        # relevant_chunk = "\n\n".join([chunks[i].page_content for i in relevant_chunk_indices])
        # context =relevant_chunk
        # context = " ".join(relevant_texts)
        # response = ai21.complete(query=query, context=context, maxTokens=150, topKReturn=1, temperature=0.1)
        response = chain.run(context=context, question=query)
        st.write(response)

if __name__ == "__main__":
    app()