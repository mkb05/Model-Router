import os
import json
import time
import random
from groq import Groq

# Ensure your GROQ_API_KEY is available
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

TAXONOMY = [
    {
        "domain": "code_engineering",
        "simple_desc": "boilerplate code, syntax fixes, basic regex, writing docstrings, converting json to dict",
        "complex_desc": "concurrency issues, lock-free data structures, distributed consensus, memory leak profiling, kernel bypass"
    },
    {
        "domain": "math_logic",
        "simple_desc": "basic arithmetic, linear algebra definitions, unit conversion, simple probability formulas",
        "complex_desc": "formal mathematical induction proofs, discrete optimization, constraint satisfaction puzzles, game theory equilibria"
    },
    {
        "domain": "architecture_systems",
        "simple_desc": "standard CRUD design, connecting db to backend, basic dockerfile, simple REST endpoints",
        "complex_desc": "event-driven architectures under high contention, multi-region replication tradeoffs, disaster recovery consensus, zero-downtime database sharding"
    },
    {
        "domain": "nlp_reasoning",
        "simple_desc": "spelling/grammar checks, translation, extracting dates from text, sentiment classification",
        "complex_desc": "multi-hop causal deduction, identifying subtle logical fallacies across lengthy arguments, complex literary subtext analysis"
    }
]

SYSTEM_PROMPT = """You are a machine learning dataset synthesis engine.
Generate realistic, diverse user prompts submitted to an AI assistant.
Respond ONLY with a JSON object containing a 'prompts' key which is a list of objects with fields:
- 'text': the realistic prompt string (varied length, 5 to 60 words)
- 'label': integer (0 or 1)
No additional commentary, no markdown wrapping."""

def generate_batch(domain: str, desc: str, label: int, count: int = 15):
    prompt_instruction = f"""
    Domain: {domain}
    Target Tier: {'SIMPLE (Label 0)' if label == 0 else 'COMPLEX (Label 1)'}
    Category Details: {desc}
    
    Generate {count} unique, realistic user prompts matching this criteria. 
    Ensure diverse phrasing, tone, and formatting (questions, commands, code snippets).
    Set 'label' strictly to {label}.
    """
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_instruction}
                ],
                response_format={"type": "json_object"},
                temperature=0.75
            )
            data = json.loads(response.choices[0].message.content)
            # Normalize response structure
            items = data.get("prompts", list(data.values())[0] if isinstance(data, dict) else [])
            return [{"text": item["text"].strip(), "label": label, "domain": domain} for item in items if "text" in item]
        except Exception as e:
            print(f"[!] Warning: Batch generation attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return []

def main():
    all_data = []
    print("[*] Starting dataset synthesis...")

    # Iterate through domains and synthesize batches
    for tax in TAXONOMY:
        print(f"[*] Processing domain: {tax['domain']}")
        
        # Simple batches (label 0)
        for _ in range(4):
            batch = generate_batch(tax["domain"], tax["simple_desc"], label=0, count=15)
            all_data.extend(batch)
            time.sleep(1.2)  # Respect RPM limits
            
        # Complex batches (label 1)
        for _ in range(4):
            batch = generate_batch(tax["domain"], tax["complex_desc"], label=1, count=15)
            all_data.extend(batch)
            time.sleep(1.2)

    # Deduplicate based on prompt text
    unique_data = {item["text"]: item for item in all_data}.values()
    dataset = list(unique_data)
    random.shuffle(dataset)

    print(f"[✓] Generated {len(dataset)} distinct samples.")

    # 80/20 Train/Test split
    split_index = int(len(dataset) * 0.8)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for row in train_set:
            f.write(json.dumps(row) + "\n")

    with open("test.jsonl", "w", encoding="utf-8") as f:
        for row in test_set:
            f.write(json.dumps(row) + "\n")

    print(f"[✓] Saved {len(train_set)} rows to train.jsonl")
    print(f"[✓] Saved {len(test_set)} rows to test.jsonl")

if __name__ == "__main__":
    main()