import requests
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# Function to get response from the bot and measure time
def get_bot_response_with_time(question):
    url = 'http://127.0.0.1:5000/ask'
    start_time = time.time()
    response = requests.post(url, data={'query': question})
    end_time = time.time()
    
    if response.status_code == 200:
        try:
            response_json = response.json()
            if isinstance(response_json, dict) and 'response' in response_json:
                response_text = response_json['response']
            elif isinstance(response_json, str):
                response_text = response_json
            else:
                response_text = ''
        except ValueError:
            response_text = ''
    else:
        response_text = ''
    
    elapsed_time = end_time - start_time
    return response_text, elapsed_time

# Define the questions with increasing complexity
questions = {
    'Simple': [
        "What is gestational diabetes?",
        "What are the symptoms of gestational diabetes?",
        "How is gestational diabetes diagnosed?",
        "What are the risk factors for gestational diabetes?",
        "How is gestational diabetes managed during pregnancy?"
    ],
    'One Task One Context': [
        "What is gestational diabetes?",
        "How can gestational diabetes be detected?",
        "What are the common treatments for gestational diabetes?",
        "What lifestyle changes help manage gestational diabetes?",
        "How does gestational diabetes affect delivery?"
    ],
    'Multi Tasks One Context': [
        "What are the symptoms and risk factors for gestational diabetes?",
        "Describe the management and treatment options for gestational diabetes.",
        "How is gestational diabetes diagnosed and what are its impacts on pregnancy?",
        "Discuss dietary recommendations and their effects on managing gestational diabetes.",
        "Explain the impact of gestational diabetes on both the mother and the baby, and discuss preventive measures."
    ],
    'Multi Tasks Multi Contexts': [
        "Compare gestational diabetes with type 2 diabetes in terms of risk factors and management.",
        "Analyze the dietary changes recommended for gestational diabetes and their long-term effects on maternal health.",
        "Discuss the implications of gestational diabetes on labor and delivery compared to other pregnancy complications.",
        "Evaluate recent research on gestational diabetes and its effects on prenatal care and infant health.",
        "Assess different treatment options for gestational diabetes and their impact on long-term maternal and fetal outcomes."
    ]
}

# Measure response times and prepare data
data = []
for complexity, qs in questions.items():
    for i, question in enumerate(qs):
        _, elapsed_time = get_bot_response_with_time(question)
        data.append({'Complexity': complexity, 'Question': f'Q{i+1}', 'Response Time (s)': elapsed_time})

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting the box plot
plt.figure(figsize=(14, 8))
sns.boxplot(x='Complexity', y='Response Time (s)', data=df, palette="Set2")
plt.title('Response Times for Different Question Complexities')
plt.grid(True)
plt.show()
