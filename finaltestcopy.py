import requests
import numpy as np
import pandas as pd
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from rouge_score import rouge_scorer
import os
from pathlib import Path

# Function to get response from the bot
def get_bot_response(question):
    url = 'http://127.0.0.1:5000/ask'
    response = requests.post(url, data={'query': question})
    
    if response.status_code == 200:
        try:
            response_json = response.json()
            if isinstance(response_json, dict) and 'response' in response_json:
                return response_json['response']
            elif isinstance(response_json, str):
                return response_json
            else:
                print("Unexpected response format")
                return ''
        except ValueError:
            print("Error decoding JSON response")
            return ''
    else:
        print(f"Failed to get a valid response. Status code: {response.status_code}")
        return ''

# Function to calculate consistency
def calculate_consistency(prompt, response, model):
    embeddings = model.encode([prompt, response])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Function to calculate relevance (similar to consistency)
def calculate_relevance(prompt, response, model):
    return calculate_consistency(prompt, response, model)

# Function to calculate novelty
def calculate_novelty(response, reference_responses, model):
    embeddings = model.encode([response] + reference_responses)
    response_embedding = embeddings[0]
    reference_embeddings = embeddings[1:]
    similarities = cosine_similarity([response_embedding], reference_embeddings)
    novelty = 1 - max(similarities[0])  # Lower similarity indicates higher novelty
    return novelty

# Function to detect hate speech
def detect_hate_speech(text):
    hate_speech_keywords = ['hate', 'violence', 'discrimination', 'racism', 'bigotry', 'prejudice']
    for keyword in hate_speech_keywords:
        if keyword in text.lower():
            return "Hate Speech Detected"
    return "Not Hate Speech"

# Function to detect bias
def detect_bias(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Simplified bias detection

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes

# Function to measure CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)  # percentage

# Function to measure resources
def measure_resources(func, *args):
    start_time = time.time()
    start_memory = get_memory_usage()

    # Call the function
    func(*args)

    end_memory = get_memory_usage()
    end_time = time.time()

    memory_used = (end_memory - start_memory) / (1024 * 1024)  # Convert to MB
    execution_time = end_time - start_time
    cpu_usage = get_cpu_usage()

    return execution_time, memory_used, cpu_usage

# Function to calculate ROUGE score
def calculate_rouge_score(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Initialize model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define prompts
prompts = [
    "When should cesarean delivery be considered for GDM?",
    "How should GDM and type 2 diabetes in pregnancy be managed?",
    "How does a history of shoulder dystocia impact future pregnancies?",
    "How can lifestyle changes prevent type 2 diabetes in women with a history of GDM?",
    "What are the key findings on diagnosing GDM?"
]

# Define reference answers (you can use your own set)
reference_answers = [
    "Cesarean delivery should be considered for GDM if there are concerns about fetal macrosomia, if the baby is too large for a vaginal delivery, or if there are complications such as preeclampsia or abnormal fetal heart rate patterns.",
    "GDM and type 2 diabetes in pregnancy should be managed through a combination of lifestyle changes (diet and exercise), blood glucose monitoring, and medication if necessary. Insulin or oral hypoglycemic agents may be used if lifestyle changes alone are insufficient to control blood sugar levels.",
    "A history of shoulder dystocia can increase the risk of recurrence in future pregnancies. Management may involve close monitoring, planning for possible cesarean delivery, and considering the use of different delivery techniques to minimize the risk of shoulder dystocia.",
    "Lifestyle changes such as maintaining a healthy diet, engaging in regular physical activity, and achieving and maintaining a healthy weight can help prevent type 2 diabetes in women with a history of GDM. Regular monitoring and managing blood sugar levels are also important.",
    "Key findings in diagnosing GDM include elevated blood glucose levels during pregnancy, typically detected through screening tests such as the oral glucose tolerance test (OGTT). Diagnosis is confirmed if blood glucose levels exceed specified thresholds during the test."
]

# Store results
results = []

for prompt in prompts:
    response = get_bot_response(prompt)
    time_taken, memory_used, cpu_usage = measure_resources(get_bot_response, prompt)

    consistency = calculate_consistency(prompt, response, model)
    relevance = calculate_relevance(prompt, response, model)
    novelty = calculate_novelty(response, reference_answers, model)
    hate_speech = detect_hate_speech(response)
    bias = detect_bias(response)
    rouge = calculate_rouge_score(prompt, response)

    results.append({
        'Prompt': prompt[:50] + ('...' if len(prompt) > 50 else ''),  # Truncate long prompts
        'Response': response[:50] + ('...' if len(response) > 50 else ''),  # Truncate long responses
        'Consistency': consistency,
        'Relevance': relevance,
        'Novelty': novelty,
        'Hate Speech': hate_speech,
        'Bias': bias,
        'Time (s)': time_taken,
        'Memory (MB)': memory_used,
        'CPU Usage (%)': cpu_usage,
        'ROUGE1': rouge['rouge1'].fmeasure,
        'ROUGE2': rouge['rouge2'].fmeasure,
        'ROUGEL': rouge['rougeL'].fmeasure
    })

# Create a DataFrame
df_results = pd.DataFrame(results)

# Define output directories
project_dir = 'visualizations/'
downloads_dir = str(Path.home() / 'Downloads' / 'visualizations')

# Create output directories if they don't exist
os.makedirs(project_dir, exist_ok=True)
os.makedirs(downloads_dir, exist_ok=True)

# Function to save individual plots
def save_plot(data, plot_type, x_col=None, y_col=None, title=None, xlabel=None, ylabel=None, filename=None):
    plt.figure(figsize=(10, 6))
    if plot_type == 'bar':
        sns.barplot(x=x_col, y=y_col, data=data)
    elif plot_type == 'count':
        sns.countplot(x=y_col, data=data)
    elif plot_type == 'hist':
        sns.histplot(data[y_col], bins=20)
    elif plot_type == 'line':
        for metric in y_col:
            plt.plot(data.index, data[metric], label=metric)
        plt.legend()
    if title:
        plt.title(title, fontsize=14)
    if xlabel:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotate x-axis labels for better alignment
    plt.tight_layout()
    if filename:
        project_filepath = os.path.join(project_dir, filename)
        downloads_filepath = os.path.join(downloads_dir, filename)
        plt.savefig(project_filepath)
        plt.savefig(downloads_filepath)
    plt.close()

# Plot and save Consistency
save_plot(df_results, 'bar', 'Prompt', 'Consistency', 'Consistency of Responses', 'Prompt', 'Consistency', 'consistency_plot.png')

# Plot and save Relevance
save_plot(df_results, 'bar', 'Prompt', 'Relevance', 'Relevance of Responses', 'Prompt', 'Relevance', 'relevance_plot.png')

# Plot and save Novelty
save_plot(df_results, 'bar', 'Prompt', 'Novelty', 'Novelty of Responses', 'Prompt', 'Novelty', 'novelty_plot.png')

# Plot and save Hate Speech Detection
save_plot(df_results, 'count', None, 'Hate Speech', 'Hate Speech Detection', 'Hate Speech', 'Count', 'hate_speech_plot.png')

# Plot and save Bias Detection
save_plot(df_results, 'hist', None, 'Bias', 'Bias Detection', 'Bias', 'Frequency', 'bias_plot.png')

# Plot and save ROUGE Scores
save_plot(df_results, 'line', None, ['ROUGE1', 'ROUGE2', 'ROUGEL'], 'ROUGE Scores', 'Index', 'Score', 'rouge_scores_plot.png')

# Plot and save Time Taken
save_plot(df_results, 'bar', 'Prompt', 'Time (s)', 'Time Taken for Responses', 'Prompt', 'Time (s)', 'time_taken_plot.png')

# Plot and save CPU Usage
save_plot(df_results, 'bar', 'Prompt', 'CPU Usage (%)', 'CPU Usage During Responses', 'Prompt', 'CPU Usage (%)', 'cpu_usage_plot.png')

# Plot and save Memory Used
save_plot(df_results, 'bar', 'Prompt', 'Memory (MB)', 'Memory Used During Responses', 'Prompt', 'Memory (MB)', 'memory_used_plot.png')

print("All visualizations have been saved in both the project directory and the Downloads folder.")
