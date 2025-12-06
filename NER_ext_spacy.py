import pandas as pd
import spacy
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
print("Columns in dataset:", df.columns)

# Create sets to store unique entities
persons, orgs, dates = set(), set(), set()

# Process each sentence
for text in df['sentence']:
    doc = nlp(str(text))
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.add(ent.text)
        elif ent.label_ == "ORG":
            orgs.add(ent.text)
        elif ent.label_ == "DATE":
            dates.add(ent.text)

# Display results
print("Unique PERSON entities:", len(persons))
print("Unique ORG entities:", len(orgs))
print("Unique DATE entities:", len(dates))

# Create output directory
output_dir = "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed"
os.makedirs(output_dir, exist_ok=True)

# Save each set separately to CSVs
pd.DataFrame({"PERSON": list(persons)}).to_csv(os.path.join(output_dir, "persons.csv"), index=False)
pd.DataFrame({"ORG": list(orgs)}).to_csv(os.path.join(output_dir, "orgs.csv"), index=False)
pd.DataFrame({"DATE": list(dates)}).to_csv(os.path.join(output_dir, "dates.csv"), index=False)

print(" Entities saved to individual CSV files in:", output_dir)
