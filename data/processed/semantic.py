import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load dataset

triples_path = "/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner_triples.csv"
df = pd.read_csv(triples_path)
print("✅ File loaded successfully!")
print("Columns:", df.columns)
print(df.head())

# Create natural-language sentences

sample_df = df.head(10)

sentences = []
for _, row in sample_df.iterrows():
    subj = str(row["Subject"])
    rel = str(row["Relation"]).replace("_", " ")
    obj = str(row["Object"])
    sentences.append(f"{subj} is {rel} {obj}.")

print("\n✅ Natural-language sentences generated from triples:")
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}")

#  Load semantic model

print("\n⏳ Loading semantic model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

#  Semantic search query

query = input("\n🔍 Enter your query (e.g., 'Who is part of a group?' or 'Political leaders in UK'): ")
query_embedding = model.encode(query, convert_to_tensor=True)

cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

# Get top 3 most similar sentences
top_results = sorted(
    list(enumerate(cosine_scores)),
    key=lambda x: x[1],
    reverse=True
)[:3]

print("\n🔎 Top 3 most semantically similar sentences:")
for idx, score in top_results:
    print(f"👉 {sentences[idx]}  (Score: {score:.4f})")

# Reflection Prompt

print("\n📝 Reflection:")
print("- What kinds of queries gave the most accurate matches?")
print("- Where did the model make mistakes?")
print("- How could you improve the quality of your triples or embeddings?")
