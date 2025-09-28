import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

device = 'cpu'
print(f"Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="gval0/NLP-Final-Georgian-Text-Embeddings-Fine-Tuned",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

FAISS_INDEX_PATH = "faiss_index"

def preprocess_and_save_faiss(file_path="georgian-civil-code.pdf", is_pdf=True):
    if os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index already exists. Skipping preprocessing.")
        return

    if is_pdf:
        loader = PyMuPDFLoader(file_path)
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    all_splits = []
    for doc in docs:
        articles = re.split(r'(მუხლი \d+)', doc.page_content)[1:]
        for i in range(0, len(articles), 2):
            article_title = articles[i].strip()
            article_text = articles[i+1].strip()

            article_text = re.sub(r'\s+', ' ', article_text)
            article_text = article_text.replace(' .', '.').replace(' ,', ',')
            article_text = re.sub(r'(\w+)-(\w+)', r'\1\2', article_text)

            article_number = re.search(r'\d+', article_title).group(0) if re.search(r'\d+', article_title) else 'unknown'
            all_splits.append(Document(
                page_content=article_text,
                metadata={"article": article_number}
            ))

    embedding_dim = len(embeddings.embed_query("საქართველოს სამოქალაქო კოდექსის RAG აგენტი"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(all_splits)

    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")


preprocess_and_save_faiss("georgian-civil-code.pdf")