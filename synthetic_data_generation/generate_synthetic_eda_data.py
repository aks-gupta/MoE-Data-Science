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

DOMAINS = ["finance", "medical", "ecommerce", "climate", "education",
           "social_media", "real_estate", "transportation", "retail",
           "manufacturing", "telecommunications", "energy", "agriculture",
           "government", "entertainment"]

# ----------------------------
#  Synthetic EDA Prompt
# ----------------------------
GEN_PROMPT = """
You are an expert in exploratory data analysis and EDA summary generation.

Your job is to generate a NEW synthetic EXAMPLE in the [DOMAIN] domain that imitates EDA summary datasets.

You MUST return a JSON object with EXACTLY these fields:

{
    "purpose": "...",
    "raw_table": "...CSV string...",
    "eda_steps": [... list of EDA instructions ...],
    "eda_results": {... dictionary of computed outputs ...}
}

-------------------------------------------
RULES FOR GENERATING THE EXAMPLE:
-------------------------------------------

1. PURPOSE
 - 1–2 sentences.
 - Examples: "Analyze product sales trends", "Identify most common diagnoses", etc.

2. RAW TABLE
 - MUST be 8–15 rows.
 - MUST be a CSV string.
 - MUST contain some messy or realistic values:
        * inconsistent capitalization
        * missing values
        * mixed types
        * unusual categories
        * inconsistent date formatting
 - Columns should be domain-appropriate.

3. EDA STEPS  (5–10 steps)
 - MUST be a JSON list.
 - Examples of steps:
       * "Compute descriptive statistics for numeric columns"
       * "Generate value counts for categorical columns"
       * "Check missing value percentages"
       * "Compute correlation matrix"
       * "Identify top-N categories"
       * "Summarize distribution skewness"
 - MUST be written as simple English instructions.

4. EDA RESULTS
 - MUST be a JSON dictionary.
 - MUST include realistic EDA outputs such as:
        * {"summary_stats": {...}}
        * {"value_counts": {...}}
        * {"missing_values": {...}}
        * {"correlations": {...}}
        * {"top_categories": {...}}
 - Numbers must match the raw_table logically.

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

            # Validate JSON
            example = json.loads(content)

            if not all(k in example for k in ["purpose", "raw_table", "eda_steps", "eda_results"]):
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
def generate_synthetic_eda(n=100, outfile="synthetic_eda.jsonl"):
    print(f"\n🚀 Generating {n} synthetic EDA examples...\n")

    with open(outfile, "w") as f:
        for i in range(n):
            domain = random.choice(DOMAINS)  # limit to first 5 for variance
            print(f"Generating example {i+1}/{n} in domain '{domain}'...")

            ex = generate_one_example(domain=domain)
            if ex is None:
                print(f"❌ Failed to generate example {i+1}/{n}")
                continue

            f.write(json.dumps(ex) + "\n")

            print(f"✔ Generated example {i+1}/{n}")
            time.sleep(0.25)

    print(f"\n🎉 Done! Saved synthetic dataset → {outfile}")


# ----------------------------
#  CLI
# ----------------------------
if __name__ == "__main__":
    generate_synthetic_eda(n=300, outfile="synthetic_eda_akshita.jsonl")
