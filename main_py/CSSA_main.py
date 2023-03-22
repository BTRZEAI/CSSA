import io
import requests
import PyPDF2
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("NLPScholars/Roberta-Earning-Call-Transcript-Classification")
model = AutoModelForSequenceClassification.from_pretrained("NLPScholars/Roberta-Earning-Call-Transcript-Classification")

# Define the labels for the model

labels = ['Uncertainty', 'Litigious', 'Positive', 'Constraining', 'Negative']


# Define a function to read in a PDF file from a URL and extract its text content
def extract_text_from_pdf(url):
    response = requests.get(url)
    with io.BytesIO(response.content) as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


# Define a function to classify the sentiment of a text string and return the logits
def get_sentiment_logits(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    logits_sent = outputs.logits.detach().numpy()[0]
    return logits_sent


# Define a function to get the sentiment logits for all PDF files in a GitHub repository
def get_sentiment_logits_for_all_files_in_github_repo(owner, repo, path):
    api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    response = requests.get(api_url)
    contents = response.json()
    logits_dicti = {}
    for file in contents:
        if file['type'] == 'file' and file['name'].endswith('.pdf'):
            url = file['download_url']
            text = extract_text_from_pdf(url)
            logits_sen = get_sentiment_logits(text)
            # Extract the date from the filename and create a datetime object
            date_str = file['name'][:8]
            date = datetime.datetime.strptime(date_str, '%Y%m%d')
            # Add a new key to the logits dictionary with the date as the value
            logits_dicti[file['name']] = {'date': date.date(), 'logits': logits_sen}

    return logits_dicti


# Example usage
logits_dict = get_sentiment_logits_for_all_files_in_github_repo('BTRZEAI', 'CSSA', 'lib\pdf_files')

# Print the logits for each PDF file
for filename, logits in logits_dict.items():
    print(f"{filename}:")
    for label, logit in zip(labels, logits):
        print(f"{label}: {logit}")
    print()
