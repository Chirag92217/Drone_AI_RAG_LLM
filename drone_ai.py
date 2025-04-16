# main.py

from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models import LLM
from langchain.chains import RetrievalQA
from typing import Optional, List
from pydantic import BaseModel, Field
import requests

# Load text document
file_path = 'dataa.txt'
loader = TextLoader(file_path)
document = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(document)

# Generate embeddings
embedding_obj = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup ChromaDB
persist_directory = "db_folder"
vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_obj, persist_directory=persist_directory)
vector_db.persist()
vector_db = None  # clear from memory
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_obj)

# Create retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# Custom LLM wrapper
class OllamaLLM(LLM, BaseModel):
    api_url: str = Field(..., description="URL of the hosted Ollama API")
    model_name: str = Field(default="llama3.1:8b", description="Model name")
    system_prompt: str = Field(
        default="You are a helpful Drone Trainer. Keep answers simple and concise.",
        description="System-level instruction for the model"
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


# Instantiate the custom LLM
api_url = "http://115.241.186.203/api/generate"
llm_obj = OllamaLLM(api_url=api_url, system_prompt="You are a knowledgeable and friendly drone expert and trainer. Give Clear and concise explanation to a beginner in simple words. Try to give response in fewer words")

# QA chain setup
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_obj,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# Output function
def process_llm_response(llm_response):
    print("LLM_Response >>>>>>\n", llm_response["result"])
    print("\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


# Ask questions
if __name__ == "__main__":
    questions = [
        "What if battery fails?"
    ]
    for query in questions:
        print(f"\n\nQuestion: {query}")
        response = qa_chain.invoke(query)
        process_llm_response(response)
