import json
from transformers import AutoTokenizer

# Load the JSON data
with open('sources/provisions.json', 'r') as f:
    data = json.load(f)

# Initialize a modern BERT tokenizer (using bert-base-uncased as example)
# You can also use 'bert-large-uncased', 'roberta-base', or other models
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Find the term with the most tokens in its raw field
max_tokens = 0
max_term = None
max_raw_text = None

for term, content in data.items():
    raw_text = content['raw']
    # Tokenize and count tokens
    tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    token_count = len(tokens)
    
    if token_count > max_tokens:
        max_tokens = token_count
        max_term = term
        max_raw_text = raw_text

# Display results
print(f"Term with most tokens: {max_term}")
print(f"Token count: {max_tokens}")
print(f"\nRaw text:\n{max_raw_text}")

# Show top 5 terms by token count
print("\n" + "="*80)
print("Top 5 terms by token count (BERT tokenizer):")
print("="*80)

token_counts = []
for term, content in data.items():
    raw_text = content['raw']
    tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    token_count = len(tokens)
    token_counts.append((term, token_count))

token_counts.sort(key=lambda x: x[1], reverse=True)

for i, (term, count) in enumerate(token_counts, 1):
    print(f"{i}. {term}: {count} tokens")

# Optional: Show tokenization example for the longest entry
print("\n" + "="*80)
print("Sample tokenization for longest entry:")
print("="*80)
tokens = tokenizer.tokenize(max_raw_text)
print(f"First 50 tokens: {tokens[:50]}")
print(f"Total unique tokens: {len(set(tokens))}")