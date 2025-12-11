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
#  Synthetic Feature Engineering Prompt
# ----------------------------
GEN_PROMPT = """
You are an expert in machine learning feature engineering and data transformation workflows.

Your job is to generate a NEW synthetic EXAMPLE in the [DOMAIN] domain that imitates
a feature engineering dataset.

You MUST return a JSON object with EXACTLY these fields:

{
    "purpose": "...",
    "raw_table": "...CSV string...",
    "fe_steps": [... list of feature engineering steps ...],
    "fe_table": "...CSV string..."
}

-------------------------------------------
RULES FOR GENERATING THE EXAMPLE:
-------------------------------------------

1. PURPOSE
 - 1–2 sentences describing why features are being engineered.
 - Examples:
     * "Create model-ready features for predicting delivery time."
     * "Engineer numerical and categorical features for credit risk scoring."

2. RAW TABLE
 - MUST be a CSV string.
 - MUST contain 8–15 rows.
 - MUST include realistic VALUES, including some messy patterns:
        * inconsistent capitalization
        * missing values
        * mixed date formats
        * categorical fields with variants
        * numeric fields with stray strings
 - Columns must be domain-appropriate.

3. fe_steps (5–12 steps)
 - MUST be a JSON list of feature engineering operations, such as:
       * "Extract day_of_week from Date column"
       * "Apply log transform to Amount"
       * "One-hot encode Category"
       * "Compute price_per_unit"
       * "Standardize numeric columns"
       * "Map severity labels to numeric scale"
       * "Bucketize Age into ranges"
 - Each step MUST be a readable English instruction (not code).

4. fe_table
 - MUST be a CSV string.
 - MUST contain:
       * All original columns
       * PLUS newly engineered columns produced by fe_steps
 - Values must logically follow from the steps.
 - No missing or inconsistent values in engineered columns.

5. OUTPUT
 - MUST be valid JSON.
 - MUST NOT include markdown or backticks.
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
            example = json.loads(content)

            if not all(k in example for k in ["purpose", "raw_table", "fe_steps", "fe_table"]):
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
def generate_synthetic_feature_engineering(n=100, outfile="synthetic_fe.jsonl"):
    print(f"\n🛠️ Generating {n} synthetic feature engineering examples...\n")

    with open(outfile, "w") as f:
        for i in range(n):
            domain = random.choice(DOMAINS)
            print(f"Generating example {i+1}/{n} in domain '{domain}'...")

            ex = generate_one_example(domain=domain)
            if ex is None:
                print(f"❌ Failed to generate example {i+1}/{n}")
                continue

            f.write(json.dumps(ex) + "\n")
            print(f"✔ Generated example {i+1}/{n}")
            time.sleep(0.25)

    print(f"\n🎉 Done! Saved synthetic feature-engineering dataset → {outfile}")


# ----------------------------
#  CLI
# ----------------------------
if __name__ == "__main__":
    generate_synthetic_feature_engineering(n=300, outfile="synthetic_fe_akshita.jsonl")
