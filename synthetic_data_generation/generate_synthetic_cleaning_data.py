import json
import os
import random
import time
from openai import OpenAI

# ----------------------------
#  OpenAI Client (CMU Gateway)
# ----------------------------
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ai-gateway.andrew.cmu.edu/"
)

DOMAINS = ["finance", "medical", "ecommerce", "climate", "education", "social_media", "real_estate", "transportation", "retail", "manufacturing", "telecommunications", "energy", "agriculture", "government", "entertainment"]
# ----------------------------
#  Synthetic Data Prompt
# ----------------------------
GEN_PROMPT = """
You are an expert in data cleaning, dirty data generation, and OpenRefine workflows.

Your job is to generate a NEW synthetic EXAMPLE in the [DOMAIN] domain that imitates the AutoDCWorkflow dataset exactly.

You MUST return a JSON object with all fields described.

STRUCTURE YOU MUST FOLLOW:

{
    "purpose": "...",
    "raw_table": "...CSV string...",
    "cleaning_workflow": [... list of OpenRefine steps ...],
    "clean_table": "...CSV string..."
}

-------------------------------------------
RULES FOR GENERATING THE EXAMPLE:
-------------------------------------------

1. PURPOSE
 - 1–2 sentences.
 - Something realistic like "Clean inconsistent city names", "Normalize business types."

2. RAW TABLE
 - MUST be 8–15 rows.
 - MUST be a CSV string.
 - MUST include messy values such as:
        * inconsistent capitalization
        * extra underscores or hyphens
        * misspellings
        * missing values
        * wrong formats
        * wrong numeric / date types
 - Columns should be realistic: City, State, BusinessType, Price, LoanAmount, Date, etc.

3. CLEANING WORKFLOW
 - MUST be a JSON list.
 - MUST contain 5–12 operations.
 - Operations must use REAL OpenRefine operation schema:
       * core/text-transform
       * core/mass-edit
       * core/date-parse
       * core/fill-down
       * core/split
       * core/column-rename
 - Each step MUST have:
        "op": "core/text-transform" (or another)
        "columnName": "...",
        "expression": "...",
        plus any needed args
 - mass-edit steps MUST include:
        "edits": [{"from": [...], "to": "..."}]

4. CLEANED TABLE
 - Apply the workflow logically.
 - Same number of rows and columns as raw table.
 - All messy values must be cleaned.

5. OUTPUT
 - MUST be valid JSON.
 - DO NOT include markdown or backticks.
"""

# ----------------------------
#  Generate One Example
# ----------------------------
def generate_one_example(domain, retries=3):
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0.8,
                messages=[{"role": "user", "content": GEN_PROMPT.replace("[DOMAIN]", domain)}]
            )

            content = response.choices[0].message.content.strip()

            # Try parsing JSON
            example = json.loads(content)

            # Mandatory keys check
            if not all(k in example for k in ["purpose", "raw_table", "cleaning_workflow", "clean_table"]):
                raise ValueError("Missing required keys.")

            return example

        except Exception as e:
            print("Retry due to error:", e)
            print("Raw model output:\n", content)
            time.sleep(1.0)

    return None


# ----------------------------
#  Generate N Examples
# ----------------------------
def generate_synthetic_autodc(n=100, outfile="synthetic_autodc.jsonl"):
    print(f"\n🚀 Generating {n} synthetic AutoDC examples...\n")

    with open(outfile, "w") as f:
        for i in range(n):
            domain = random.choice(DOMAINS[:5]) # limit to first 5 domains for diversity
            print(f"Generating example {i+1}/{n} in domain '{domain}'...")
            ex = generate_one_example(domain=domain)
            if ex is None:
                print(f"❌ Failed to generate example {i+1}/{n}")
                continue

            f.write(json.dumps(ex) + "\n")

            print(f"✔ Generated example {i+1}/{n}")
            time.sleep(0.25)   # prevent rate limit

    print(f"\n🎉 Done! Saved synthetic dataset → {outfile}")


# ----------------------------
#  CLI
# ----------------------------
if __name__ == "__main__":
    generate_synthetic_autodc(n=100, outfile="synthetic_autodc_akshita_3.jsonl")
