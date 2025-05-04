from document_parser import DocumentParser
from retriever import FaissRetriever
from inference import FLANQAInferencer
from config import EMBEDDING_MODEL
import os

def test_system():
    # Initialize components
    parser = DocumentParser()
    retriever = FaissRetriever()
    inferencer = FLANQAInferencer()
    
    # Process test documents
    print("\n📄 Processing test documents...")
    for doc_file in ["test_document.txt", "test_document2.txt"]:
        print(f"\nProcessing {doc_file}...")
        result = parser.process_document(doc_file)
        retriever.add_documents(result['chunks'], result['embeddings'])
    
    # Interactive testing
    while True:
        print("\n❓ Enter a question (or type 'exit' to quit):")
        question = input()
        
        if question.lower() == 'exit':
            break
            
        print(f"\n🔍 Finding relevant context...")
        context = retriever.search(question, top_k=3)
        context_text = " ".join([c['text'] for c in context])
        
        print("\n📝 Generating answer...")
        answer = inferencer.answer(question, context_text)
        
        print("\nAnswer:")
        print(answer)
        
        print("\nContext used:")
        for i, chunk in enumerate(context):
            print(f"\nChunk {i+1} (Score: {chunk['score']:.2f}):")
            print(chunk['text'])

if __name__ == "__main__":
    test_system()
