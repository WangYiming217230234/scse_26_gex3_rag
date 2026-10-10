from rag_tool import get_retriever
from agent import run_agent

def main():
    print("=== University IT Support RAG System ===")
    retriever = get_retriever()
    
    question = "What exact phone number should I call for the University IT Service Desk?"
    print(f"\nUser Question: {question}")
    
    context = retriever(question, k=3)
    
    print("\n=== Agent Processing ===")
    final_answer = run_agent(question, context)
    
    print("\n=== FINAL ANSWER ===")
    print(final_answer)

if __name__ == "__main__":
    main()