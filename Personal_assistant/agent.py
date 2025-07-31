from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import config

def load_chat_agent():
    db = FAISS.load_local(
        "vectorstore",
        OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY),
        allow_dangerous_deserialization=True  # 👈 Add this line
    )

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name=config.LLM_MODEL, temperature=0, openai_api_key=config.OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    return qa_chain

