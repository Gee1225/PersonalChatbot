import streamlit as st
from agent import load_chat_agent

st.set_page_config(page_title="Personal Chatbot", layout="centered")
st.title("🤖 Ask Me Anything – Personal Chatbot")

# Load the QA chain
qa = load_chat_agent()

# User input
user_input = st.text_input("Ask something about George(Use 'George'):")

if user_input:
    with st.spinner("Thinking..."):
        response = qa.invoke({"query": user_input})

        # Display answer
        st.markdown("### 💬 Answer")
        st.write(response["result"])

        # Show source documents (optional)
        if "source_documents" in response:
            with st.expander("📝 Source Documents"):
                for i, doc in enumerate(response["source_documents"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(doc.page_content)
