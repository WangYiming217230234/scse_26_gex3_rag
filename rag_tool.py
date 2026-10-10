import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from policy_loader import load_policy_documents

PERSIST_DIR = "./chroma_db"

def get_retriever():
    print("Loading policies and initializing vector store...")
    documents = load_policy_documents()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = Chroma(
        collection_name="university_support",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    if vector_store._collection.count() == 0:
        print(f"Adding {len(documents)} documents to Chroma database...")
        vector_store.add_documents(documents)
    else:
        print("Chroma database already populated.")

    def retrieve(question, k=3):
        print(f"-> Retrieving context for: '{question}'")
        results = vector_store.similarity_search(question, k=k)
        
        context = ""
        for i, doc in enumerate(results):
            context += f"--- Evidence {i+1} ---\n{doc.page_content}\n\n"
            
        if not context.strip():
            return "No relevant policies found in the provided knowledge base."
            
        return context.strip()

    return retrieve

if __name__ == "__main__":
    retriever = get_retriever()
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    res = retriever(test_question, k=3)
    
    print("\n=== RETRIEVAL RESULTS ===")
    print(res)
    print("=========================")