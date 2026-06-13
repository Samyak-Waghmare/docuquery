import importlib.metadata as md

pkgs = ['streamlit', 'langchain', 'langchain-community', 'langchain-text-splitters',
        'langchain-qdrant', 'langchain-google-genai', 'openai', 'google-generativeai',
        'qdrant-client', 'pypdf', 'python-dotenv', 'pillow']
print('=== VERSIONS ===')
for p in pkgs:
    try:
        print(f'{p}=={md.version(p)}')
    except Exception:
        print(f'# {p}  -> NOT INSTALLED')

print('\n=== IMPORT TEST (what app.py actually imports) ===')
mods = ['streamlit', 'dotenv', 'PIL.Image', 'pypdf',
        'langchain_community.document_loaders', 'langchain_text_splitters',
        'langchain_qdrant', 'langchain_google_genai', 'openai', 'qdrant_client']
for m in mods:
    try:
        __import__(m)
        print(f'ok   {m}')
    except Exception as e:
        print(f'FAIL {m}  -> {type(e).__name__}: {str(e)[:90]}')

print('\n=== SPECIFIC SYMBOLS app.py uses ===')
checks = [
    ('langchain_community.document_loaders', 'PyPDFLoader'),
    ('langchain_text_splitters', 'RecursiveCharacterTextSplitter'),
    ('langchain_qdrant', 'QdrantVectorStore'),
    ('langchain_google_genai', 'GoogleGenerativeAIEmbeddings'),
    ('openai', 'OpenAI'),
    ('qdrant_client', 'QdrantClient'),
]
for mod, sym in checks:
    try:
        m = __import__(mod, fromlist=[sym])
        getattr(m, sym)
        print(f'ok   {mod}.{sym}')
    except Exception as e:
        print(f'FAIL {mod}.{sym}  -> {type(e).__name__}: {str(e)[:90]}')
