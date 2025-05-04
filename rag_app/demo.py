from document_parser import DocumentParser
from retriever import FaissRetriever
from inference import FLANQAInferencer
from config import UPLOAD_FOLDER
import os
import json

def main():
    # Initialize components
    parser = DocumentParser()
    retriever = FaissRetriever()
    inferencer = FLANQAInferencer()
    
    # Process both test documents
    print("\n📄 Processing test documents...")
    for doc_file in ["test_document.txt", "test_document2.txt"]:
        print(f"\nProcessing {doc_file}...")
        result = parser.process_document(doc_file)
        retriever.add_documents(result['chunks'], result['embeddings'])
        
    # Test questions with different types of queries
    test_cases = [
        {
            "question": "What is AI?",
            "expected": "AI is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and other animals"
        },
        {
            "question": "What are the types of AI systems?",
            "expected": "AI can be classified into three different types of systems: analytical, human-inspired, and humanized artificial intelligence"
        },
        {
            "question": "What are some applications of ML?",
            "expected": "Applications of ML include computer vision, natural language processing, healthcare, finance, and autonomous systems"
        },
        {
            "question": "What is supervised learning?",
            "expected": "Supervised Learning uses labeled data to train models for tasks like classification and regression"
        },
        {
            "question": "What are the challenges in ML?",
            "expected": "Challenges in ML include data privacy, bias, interpretability, computation, and maintenance"
        }
    ]
    
    print("\n📝 Testing the system with various questions:")
    results = []
    
    for test in test_cases:
        print(f"\n\nQuestion: {test['question']}")
        
        # Get relevant context
        context = retriever.search(test['question'], top_k=3)
        context_text = " ".join([c['text'] for c in context])
        
        # Get answer
        answer = inferencer.answer(test['question'], context_text)
        print(f"Answer: {answer}")
        
        # Store results for analysis
        results.append({
            "question": test['question'],
            "answer": answer,
            "context": context_text,
            "retrieved_chunks": len(context)
        })
    
    # Save results for analysis
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Test results saved to test_results.json")
    print("\n🔍 To analyze the results:")
    print("1. Check the accuracy of answers")
    print("2. Review the retrieved context")
    print("3. Examine the number of chunks retrieved")
    print("4. Compare with expected answers")

if __name__ == "__main__":
    main()
