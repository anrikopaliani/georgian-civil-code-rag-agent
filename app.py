import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import os, shutil

load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",  
    model_kwargs={'device': 'cpu',  "trust_remote_code": True},
    encode_kwargs={'normalize_embeddings': True}
)



@st.cache_resource(show_spinner="მონაცემების ჩატვირთვა...")
def load_data():
    os.makedirs("faiss_index", exist_ok=True)

    faiss_path = hf_hub_download(
        repo_id="anriKo/georgian-civil-code-dataset",
        filename="index.faiss",   
        repo_type="dataset"
    )
    pkl_path = hf_hub_download(
        repo_id="anriKo/georgian-civil-code-dataset",
        filename="index.pkl",    
        repo_type="dataset"
    )

    shutil.copy(faiss_path, "faiss_index/index.faiss")
    shutil.copy(pkl_path, "faiss_index/index.pkl")

    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)



vector_store = load_data()

template = """
შენ ხარ საქართველოს სამოქალაქო კოდექსის ექსპერტი იურისტი. შენი ამოცანაა უპასუხო მომხმარებლის კითხვას ქართულ ენაზე, მხოლოდ მოწოდებული კონტექსტის საფუძველზე, გრამატიკულად სწორი და ფორმალური ქართული წინადადებებით.
ინსტრუქციები:
1. გაანალიზე კითხვა და კონტექსტი ნაბიჯ-ნაბიჯ, რათა დაადგინო ყველა რელევანტური მუხლი.
2. თუ რამდენიმე მუხლია რელევანტური, ცალ-ცალკე შეიტანე თითოეული მუხლის ციტატა და მიუთითე მისი ნომერი (მაგ., „მუხლი 1344“).
3. ყოველ ციტატას თან ახლდეს **სიტყვასიტყვით ციტატა** კოდექსიდან, გამოყოფილი **თამამი შრიფტით** მარკდაუნის ფორმატში.
4. პასუხი დაიწყე მოკლე, ზუსტი ახსნით, რომელიც აჯამებს ყველა რელევანტურ მუხლს.
5. არ გამოიყენო ან გამოიგონო ინფორმაცია, რომელიც არ არის მოცემულ კონტექსტში.
6. თუ კონტექსტში პასუხი ვერ მოიძებნა, ზუსტად დაწერე: „ამ შეკითხვაზე პასუხი სამოქალაქო კოდექსში ვერ მოიძებნა.“
მომხმარებლის კითხვა:
{question}
რელევანტური მუხლები:
{context}
პასუხი:
"""


prompt = PromptTemplate(
    template=template, input_variables=["question", "context"]
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(
        f"მუხლი {doc.metadata.get('article', '?')}:\n{doc.page_content}"
        for doc in state["context"]
    )
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": response.content
    }


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

st.set_page_config(page_title="Georgian Civil Code RAG", page_icon="📜", layout="centered")

st.title("📜 საქართველოს სამოქალაქო კოდექსის RAG აგენტი")
st.write("შეიყვანეთ კითხვა და მიიღეთ ზუსტი პასუხი კოდექსიდან.")

question = st.text_area("👉 თქვენი კითხვა:", height=100)

if st.button("🔎 მოძებნე"):
    if question.strip():
        with st.spinner("დოკუმენტების მოძიება..."):
            state = retrieve({"question": question})
        
        docs_content = "\n\n".join(
            f"მუხლი {doc.metadata.get('article', '?')}:\n{doc.page_content}"
            for doc in state["context"]
        )
        messages = prompt.invoke({"question": question, "context": docs_content})
        
        st.subheader("პასუხი:")
        placeholder = st.empty()
        accumulated_response = ""
        
        with st.spinner("პასუხის გენერაცია..."):
            response_stream = llm.stream(messages)
            first_chunk = next(response_stream)
            accumulated_response += first_chunk.content
            placeholder.markdown(accumulated_response)
        
        for chunk in response_stream:
            accumulated_response += chunk.content
            placeholder.markdown(accumulated_response)
        
    else:
        st.warning("გთხოვთ შეიყვანოთ კითხვა.")