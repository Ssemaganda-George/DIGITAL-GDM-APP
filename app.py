from flask import Flask, request, jsonify, render_template, session
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = b'\xdb\x908\x9bsKD\x1c\x91\x8a\xd84\x01\xcb\xa5]\x8b\xa9n\x10\xd7\x1e\x11g'

# Load environment variables
load_dotenv()

# Load data and embeddings
data = open('cleaned_data.txt', 'r', encoding='utf-8').readlines()
embeddings = np.load('embeddings.npy')

# Set up FAISS index
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embeddings)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to retrieve context
def retrieve_context(query, top_n=10):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    _, indices = faiss_index.search(query_embedding, top_n)
    return [data[idx] for idx in indices[0]]

# Function to generate response with OpenAI API
def generate_response(conversation_history):
    openai.api_key = os.getenv('API_KEY')
    messages = [{"role": "system", "content": "You are a helpful assistant specialized in maternal health information, with a focus on gestational diabetes. Provide accurate, concise, and informative responses based on the given context. If the question is not related to maternal health or gestational diabetes, politely inform the user that you can only provide information on maternal health and gestational diabetes."}]
    messages += [{"role": "user", "content": entry['query']} for entry in conversation_history if entry['query']]
    messages += [{"role": "assistant", "content": entry['response']} for entry in conversation_history if entry['response']]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=1500,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if request.is_json:
        user_query = request.json.get('query')
    else:
        user_query = request.form.get('query')
    
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    if 'conversation_history' not in session:
        session['conversation_history'] = []

    contexts = retrieve_context(user_query)
    combined_context = "\n\n".join(contexts)

    conversation_history = session['conversation_history']
    conversation_history.append({'query': user_query, 'response': combined_context})
    response = generate_response(conversation_history)
    conversation_history[-1]['response'] = response

    session['conversation_history'] = conversation_history
    return jsonify({'response': response, 'conversation_history': session['conversation_history']})

if __name__ == '__main__':
    app.run(debug=True)
