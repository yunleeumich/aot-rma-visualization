import pandas as pd
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
print("PyTorch version:", torch.__version__)
print("CUDA version in PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

# Replace with the csv file you desired
df = pd.read_csv('unit_analysis_nlp.csv')

# Failure description column
df['Failure Description'] = df['Failure Description'].str.replace('&nbsp;', '', regex=False).replace('Nan', 'blank').replace('& nbsp;', '').replace('-&gt;','').replace('& nbsp;','')
# Replace nan value with str to aviod error
df['Failure Description'].fillna("blank", inplace=True)

# Problem Analysis column
df['Problem Analysis'] = df['Problem Analysis'].str.replace('&nbsp;', '', regex=False).replace('Nan', 'blank').replace('& nbsp;', '').replace('-&gt;','').replace('& nbsp;','')
# Replace nan value with str to aviod error
df['Problem Analysis'].fillna("blank", inplace=True)


model_name = "Helsinki-NLP/opus-mt-de-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # Perform translation
        translated = model.generate(**inputs, max_length=512)
        # Decode the translated text
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return str(e)

# Apply the translation logic on both columns
df['Failure_Description_Translated'] = df['Failure Description'].apply(translate)
df['Problem_Analysis_Translated'] = df['Problem Analysis'].apply(translate)

df.to_csv('df_failure_translated.csv')
