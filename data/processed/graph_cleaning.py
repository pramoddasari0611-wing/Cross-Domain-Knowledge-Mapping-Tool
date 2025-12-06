import pandas as pd
import networkx as nx

# LOAD DATA

df = pd.read_csv("ner_triples.csv")

print("=== ORIGINAL DATA ===")
print(df)

# 1. IDENTIFY DUPLICATE NODES

def find_duplicate_nodes(df):
    nodes = list(df["Subject"]) + list(df["Object"])
    duplicates = pd.Series(nodes).value_counts()
    duplicates = duplicates[duplicates > 1]
    return duplicates

duplicate_nodes = find_duplicate_nodes(df)
print("\n=== DUPLICATE NODES FOUND ===")
print(duplicate_nodes)

# 2. FIND INCORRECT / LOW-VALUE RELATIONS
# Criteria:
# - meaningless relations: related_to repeated too many times
# - conflicting types: geo → gpe inconsistencies

def detect_low_value_relations(df):
    issues = []

    for _, row in df.iterrows():
        if row["Relation"] != "related_to":
            issues.append(("UNEXPECTED_REL", row))
        if row["Subject"] == row["Object"]:
            issues.append(("SELF_LOOP", row))
        if row["Subject_Type"] != row["Object_Type"] and row["Relation"] == "related_to":
            issues.append(("TYPE_CONFLICT", row))

    return issues

low_value = detect_low_value_relations(df)

print("\n=== LOW-VALUE / INCORRECT RELATIONS ===")
for issue in low_value[:10]:
    print(issue)

# 3. IDENTIFY MERGE CANDIDATES
# These are nodes that refer to same entity
# Example: British vs England vs English

merge_candidates = {
    "British": "Britain",
    "English": "Britain",
}

print("\n=== MERGE CANDIDATES ===")
print(merge_candidates)

# APPLY MERGE OPERATION

df_cleaned = df.copy()

for old, new in merge_candidates.items():
    df_cleaned["Subject"] = df_cleaned["Subject"].replace(old, new)
    df_cleaned["Object"]  = df_cleaned["Object"].replace(old, new)

print("\n=== AFTER MERGING DUPLICATES ===")
print(df_cleaned)


# 4. REMOVE GENERIC OR ORPHAN NODES
# orphan node = node that appears only once

def find_orphans(df):
    nodes = list(df["Subject"]) + list(df["Object"])
    freq = pd.Series(nodes).value_counts()
    orphans = freq[freq == 1].index.tolist()
    return orphans

orphans = find_orphans(df_cleaned)

print("\n=== ORPHAN NODES ===")
print(orphans)

# remove orphans
df_cleaned = df_cleaned[
    ~df_cleaned["Subject"].isin(orphans) &
    ~df_cleaned["Object"].isin(orphans)
]

print("\n=== AFTER REMOVING ORPHANS ===")
print(df_cleaned)

# 5. REMOVE DUPLICATE TRIPLES

df_cleaned = df_cleaned.drop_duplicates()

print("\n=== FINAL CLEANED GRAPH ===")
print(df_cleaned)


# 6. EXPORT BEFORE/AFTER

df.to_csv("graph_before_cleaning.csv", index=False)
df_cleaned.to_csv("graph_after_cleaning.csv", index=False)

print("\nSaved:")
print("- graph_before_cleaning.csv")
print("- graph_after_cleaning.csv")
