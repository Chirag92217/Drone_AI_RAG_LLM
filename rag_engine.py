# rag_engine.py

from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models import LLM
from langchain.chains import RetrievalQA
from typing import Optional, List
from pydantic import BaseModel, Field
import requests

# Load and chunk documents
file_path = "dataa.txt"
loader = TextLoader(file_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
persist_directory = "db_folder"
vector_db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)
vector_db.persist()
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Custom LLM wrapper
class OllamaLLM(LLM, BaseModel):
    api_url: str = Field(..., description="URL of the hosted Ollama API")
    model_name: str = Field(default="llama3.1:8b")
    system_prompt: str = Field(
        default="You are a helpful drone expert. Answer in simple words.",
        description="Prompt for assistant"
    )

    @property
    def _llm_type(self) -> str:
        return "ollama-custom"

    def _construct_prompt(self, user_prompt: str) -> str:
        return f"<|system|>\n{self.system_prompt}\n<|user|>\n{user_prompt.strip()}\n<|assistant|>\n"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        final_prompt = self._construct_prompt(prompt)
        payload = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, headers={"Content-Type": "application/json"}, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Request failed: {e}"

# Instantiate LLM + QA Chain
ollama_llm = OllamaLLM(api_url="http://115.241.186.203/api/generate")
qa_chain = RetrievalQA.from_chain_type(llm=ollama_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
