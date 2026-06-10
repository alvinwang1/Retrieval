import asyncio
import re
from pathlib import Path

def test_prompt_rewriting():
    prompt_path = "system_prompts/zero_shot_previous_best.txt"
    
    with open(prompt_path, "r") as f:
        system_prompt = f.read()
    
    prob_format = "✅ OUTPUT FORMAT (STRICT) - PROBABILITIES MODE\n\n" \
                 "Return ONLY a JSON array of objects. Each object must include a \"reasoning\" field and a \"probs\" field with probabilities for each class:\n\n" \
                 "[\n" \
                 "  {\n" \
                 "    \"sentence_id\": \"...\",\n" \
                 "    \"reasoning\": \"...\",\n" \
                 "    \"probs\": {\"no value\": 0.1, \"potential value\": 0.2, \"certain value\": 0.3, \"high value\": 0.4}\n" \
                 "  }\n" \
                 "]"
                 
    patterns = [
        r"✅ OUTPUT FORMAT \(STRICT\).*?(?=✅)",
        r"Return ONLY valid JSON in the following exact format:.*?Return only the JSON array\.?",
        r"Assign exactly ONE label per sentence from:.*?\[.*?\]",
        r"Assign exactly ONE label per sentence from:.*?{.*?}"
    ]
    
    modified_prompt = system_prompt
    for pattern in patterns:
        modified_prompt = re.sub(pattern, "", modified_prompt, flags=re.DOTALL)
    
    # Prepend
    modified_prompt = "### OUTPUT FORMAT (FOR PROBABILITIES MODE) - MANDATORY\n" + prob_format + "\n\n" + modified_prompt
    # Cleanup leftover label formats
    modified_prompt = re.sub(r"\[\s*{\s*\"sentence_id\".*?\"label\".*?}\s*\]", "", modified_prompt, flags=re.DOTALL)
    
    print("--- MODIFIED PROMPT START ---")
    print(modified_prompt)
    print("--- MODIFIED PROMPT END ---")

if __name__ == "__main__":
    test_prompt_rewriting()
