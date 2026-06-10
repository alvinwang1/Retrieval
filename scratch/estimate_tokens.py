import os
import json

root_dir = "sources"
total_sentences = 0
terms_count = 0

for item in os.listdir(root_dir):
    item_path = os.path.join(root_dir, item)
    if os.path.isdir(item_path):
        sentence_file = os.path.join(item_path, f"{item}-sentence.json")
        if os.path.exists(sentence_file):
            with open(sentence_file, 'r') as f:
                data = json.load(f)
                total_sentences += len(data)
                terms_count += 1

# Average Prompt Overhead
# System Prompt (V2) is ~750 tokens
# Provision is ~100-300 tokens
# 13 Sentences (Batch) are ~500-800 tokens
avg_prompt_tokens = 1500 
BATCH_SIZE = 13

total_batches = (total_sentences // BATCH_SIZE) + terms_count
estimated_input_tokens = total_batches * avg_prompt_tokens

print(f"Total Terms: {terms_count}")
print(f"Total Sentences: {total_sentences}")
print(f"Estimated Total Batches: {total_batches}")
print(f"Estimated Input Tokens: {estimated_input_tokens:,}")
print(f"Approx Cost (GPT-4o-mini @ $0.15/1M): ${estimated_input_tokens/1_000_000 * 0.15:.4f}")
print(f"Approx Cost (GPT-4o @ $2.50/1M): ${estimated_input_tokens/1_000_000 * 2.50:.4f}")
