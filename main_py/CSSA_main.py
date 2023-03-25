import io
import os
import requests
import PyPDF2
import datetime
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
    logits_dict = {}
    for content in contents:
        if content['type'] == 'file' and content['name'].endswith('.pdf'):
            url = content['download_url']
            text = extract_text_from_pdf(url)
            logits_sen = get_sentiment_logits(text)
            # Extract the date from the filename and create a datetime object
            date_str = content['name'][:8]
            date = datetime.datetime.strptime(date_str, '%Y%m%d')
            # Add a new key to the logits dictionary with the date as the value
            logits_dict[content['name']] = {'date': date.date(), 'logits': logits_sen}

    return logits_dict

# Usage
stocks = ['ABEV3', 'ALPA4', 'AMER3']
#, 'ASAI3', 'AZUL4', 'B3SA3', 'BBAS3', 'BBDC3', 'BBDC4', 'BBSE3', 'BEEF3', 'BIDI11', 'BPAC11', 'BPAN4', 'BRFS3', 'BRKM5', 'BRML3', 'CASH3', 'CCRO3', 'CIEL3', 'CMIG4', 'CMIN3', 'COGN3', 'CPFE3', 'CPLE6', 'CRFB3', 'CSAN3', 'CSNA3', 'CVCB3', 'CYRE3', 'DXCO3', 'ECOR3', 'EGIE3', 'ELET3', 'EMBR3', 'ENBR3', 'ENEV3', 'ENGI11', 'EQTL3', 'EZTC3', 'FLRY3', 'GGBR4', 'GOAU4', 'GOLL4', 'HAPV3', 'HYPE3', 'IGTI11', 'IRBR3', 'ITSA4', 'ITUB4', 'JBSS3', 'JHSF3', 'KLBN11', 'LCAM3', 'LREN3', 'LWSA3', 'MGLU3', 'MRFG3', 'MRVE3', 'MULT3', 'NTCO3', 'PCAR3', 'PETR3', 'PETR4', 'PETZ3', 'POSI3', 'PRIO3', 'QUAL3', 'RADL3', 'RAIL3', 'RDOR3', 'RENT3', 'RRRP3', 'SANB11', 'SBSP3', 'SLCE3', 'SOMA3', 'SULA11', 'SUZB3', 'TAEE11', 'TIMS3', 'TOTS3', 'UGPA3', 'USIM5', 'VALE3', 'VBBR3', 'VIIA3', 'VIVT3', 'WEGE3', 'YDUQ3']

logits_dict = {symbol: get_sentiment_logits_for_all_files_in_github_repo('BTRZEAI', 'CSSA', f'main_py/lib/pdf_files/{symbol}') for symbol in stocks}

for symbol, logits in logits_dict.items():
    print(f"Number of PDF files processed for {symbol}: {len(logits)}")

