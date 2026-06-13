import os

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI

load_dotenv()

# Gemini via OpenAI-compatible endpoint (same OpenAI SDK)
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_OPENAI_BASE)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

# Take User Query
query = input("> ")

# Vector Similarity Search [query] in DB
search_results = vector_db.similarity_search(
    query=query
)

# print("search results:", search_results)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"for result in search_results])

SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answers user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}
"""

# print("SYSTEM PROMPT", SYSTEM_PROMPT)

chat_completion = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": query },
    ]
)

print(f"🤖 : {chat_completion.choices[0].message.content}")
