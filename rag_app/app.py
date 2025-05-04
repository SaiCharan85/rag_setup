from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from document_parser import DocumentParser
from retriever import FaissRetriever
from inference import FLANQAInferencer
from config import BASE_DIR, UPLOAD_FOLDER, EMBEDDING_MODEL
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
document_parser = DocumentParser()
retriever = FaissRetriever()
inferencer = FLANQAInferencer()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a document"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Process the document
        result = document_parser.process_document(filename)
        if not result:
            return jsonify({'error': 'Failed to process document'}), 500

        # Add to retriever
        retriever.add_documents(
            result['chunks'],
            result['embeddings']
        )
        
        # Clean up uploaded file
        os.remove(filename)

        return jsonify({
            'message': 'Document processed successfully',
            'num_chunks': len(result['chunks'])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/answer', methods=['POST'])
def answer_question():
    """Answer a question using RAG"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get relevant context
        context = retriever.search(question, top_k=3)
        context_text = " ".join([c['text'] for c in context])
        
        # Generate answer
        answer = inferencer.answer(question, context_text)
        
        return jsonify({
            'answer': answer,
            'context': context_text,
            'retrieved_chunks': len(context)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
