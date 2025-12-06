import pandas as pd
import spacy
import os
nlp = spacy.load("en_core_web_sm")
data = pd.read_csv("/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner.csv")

# Normalize column names
data.columns = data.columns.str.strip().str.lower()
print("Columns in dataset:", data.columns)

# Define entity extraction function

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

#  Apply NER to each sentence

data["entities"] = data["sentence"].apply(extract_entities)
print(data.head())

#  Save processed data

output_dir = "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "entities_extracted.csv")
data.to_csv(output_path, index=False)
print(f"Entities saved to {output_path}")
