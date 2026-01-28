import json

file_path = '/Users/alvinwang/cs/Retrieval/src/sources/accommodation_trade/accommodation_trade-sentence.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
        count = len(data)
        print(f"Number of sentences: {count}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except json.JSONDecodeError:
    print(f"Error decoding JSON from file: {file_path}")
