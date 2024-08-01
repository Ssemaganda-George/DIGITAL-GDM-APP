import requests
import time
import pandas as pd
import os

# Define the test case study questions
case_study_questions = [
    "What should I do if my blood sugar levels are higher than normal during pregnancy?",
    "What are the risk factors for developing gestational diabetes?",
    "How can I manage my diet to control blood sugar levels?",
    "What medications are commonly prescribed for type 2 diabetes?",
    "How often should I monitor my blood sugar levels?",
    "What lifestyle changes can help prevent diabetes?",
    "Can diabetes be cured permanently?",
    "What are the symptoms of low blood sugar?",
    "How does exercise impact blood sugar levels?",
    "What should I do if I miss a dose of my diabetes medication?",
]

# Initialize an empty list to store the results
results = []

# Loop through each question, send it to the model, and record the response and response time
for question in case_study_questions:
    start_time = time.time()
    response = requests.post("http://127.0.0.1:5000/ask", data={'query': question})
    response_time = time.time() - start_time
    response_data = response.json()
    
    # Append the result to the results list
    results.append({
        "Question": question,
        "Response": response_data['response'],
        "Response Time (s)": response_time
    })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Determine the Downloads folder path based on the OS
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

# Save the results to a CSV file in the Downloads folder
file_path = os.path.join(downloads_folder, 'case_study_results.csv')
results_df.to_csv(file_path, index=False)

print(f"Case study results saved to '{file_path}'")
