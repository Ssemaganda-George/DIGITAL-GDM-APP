from flask import Flask, request, jsonify, render_template, session
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = b'\xdb\x908\x9bsKD\x1c\x91\x8a\xd84\x01\xcb\xa5]\x8b\xa9n\x10\xd7\x1e\x11g'  # Use your generated secret key

# Load environment variables from .env file
load_dotenv()

# Load cleaned data
def load_cleaned_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Load embeddings from file
def load_embeddings(file_path):
    return np.load(file_path)

# Initialize global variables at the module level
data = load_cleaned_data('cleaned_data.txt')
embeddings = load_embeddings('embeddings.npy')

# Ensure the correct dimension of the embeddings
embedding_dim = embeddings.shape[1]  # Dimension of the embeddings
faiss_index = faiss.IndexFlatL2(embedding_dim)  # Use L2 distance
faiss_index.add(embeddings)  # Add embeddings to the index

model = SentenceTransformer('all-MiniLM-L6-v2')

# Retrieve context using FAISS
def retrieve_context(query, faiss_index, data, top_n=10):
    start_time = time.time()
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
    D, I = faiss_index.search(query_embedding, top_n)  # Search the FAISS index
    context_retrieval_time = time.time()
    print(f"Time for context retrieval: {context_retrieval_time - start_time} seconds")
    return [data[idx] for idx in I[0]]  # I[0] since I is a list of lists

# Combine multiple contexts into a single string with clear separation
def combine_contexts(contexts):
    combined_context = "\n\n".join(contexts)
    return combined_context

# Generate response with OpenAI API
def generate_response_with_openai(conversation_history):
    start_time = time.time()
    openai.api_key = os.getenv('API_KEY')  # Load the API key from the environment variable

    # Format the conversation history for the OpenAI API
    messages = [{"role": "system", "content": "You are a helpful assistant that can handle multiple tasks and contexts in one response. Address each task separately and clearly."}]
    for entry in conversation_history:
        if entry['query']:  # Ensure there's a query before adding
            messages.append({"role": "user", "content": entry['query']})
        if entry['response']:  # Ensure there's a response before adding
            messages.append({"role": "assistant", "content": entry['response']})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=1500,  # Increase token limit if needed
        temperature=0.7
    )

    answer = response.choices[0].message['content'].strip()
    response_generation_time = time.time()
    print(f"Time for response generation: {response_generation_time - start_time} seconds")
    return answer

# Summarize the conversation history
def summarize_conversation(conversation_history):
    # Combine all responses in the conversation history
    all_responses = " ".join([entry['response'] for entry in conversation_history if entry['response']])
    
    openai.api_key = os.getenv('API_KEY')  # Load the API key from the environment variable
    summary_prompt = f"Summarize the following conversation: {all_responses}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=150,
        temperature=0.5
    )
    
    summary = response.choices[0].message['content'].strip()
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()
    user_query = request.form['query']

    # Retrieve previous context from session
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Generate context based on the current query
    contexts = retrieve_context(user_query, faiss_index, data)
    combined_context = combine_contexts(contexts)

    # Add context to the conversation history
    conversation_history = session['conversation_history']
    conversation_history.append({'query': user_query, 'response': ''})  # Add the new query to the history for context generation

    # Add the combined context as a special entry in the conversation history
    if combined_context:
        conversation_history.append({'query': '', 'response': combined_context})

    # Generate response based on the entire conversation history
    response = generate_response_with_openai(conversation_history)
    # Split response into tasks if it contains multiple tasks
    split_responses = response.split('\n\n')
    
    # Update conversation history with each task response
    for i, task_response in enumerate(split_responses):
        if i < len(conversation_history):
            conversation_history[i]['response'] = task_response
        else:
            conversation_history.append({'query': '', 'response': task_response})
    
    # Update conversation history in the session
    session['conversation_history'] = conversation_history

    # Optionally, summarize the conversation history
    summary = summarize_conversation(conversation_history)

    response_time = time.time()
    print(f"Time to get response from OpenAI: {response_time - start_time} seconds")
    print(f"Conversation History: {session['conversation_history']}")  # Debug statement to check conversation history
    
    return jsonify({'response': response, 'conversation_history': session['conversation_history'], 'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
