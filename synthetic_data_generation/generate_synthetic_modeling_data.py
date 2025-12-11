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
#  Synthetic Modeling Prompt
# ----------------------------
GEN_PROMPT = """
You are an expert in machine learning modeling, evaluation, and MLOps workflows.

Your job is to generate a NEW synthetic EXAMPLE in the [DOMAIN] domain that imitates
a modeling dataset with a complete ML workflow.

You MUST return a JSON object with EXACTLY these fields:

{
    "purpose": "...",
    "raw_table": "...CSV string...",
    "model_steps": [... list of modeling workflow steps ...],
    "model_results": {... JSON dictionary of metrics and outcomes ...}
}

-------------------------------------------
RULES FOR GENERATING THE EXAMPLE:
-------------------------------------------

1. PURPOSE  
 - 1–2 sentences describing the modeling goal.  
   Examples:
       * "Predict customer churn probability."
       * "Train a model to classify crop disease severity."
       * "Build a regression model to estimate delivery delays."

2. RAW TABLE  
 - MUST be a CSV string with 8–15 rows.  
 - MUST include:
       * numeric columns
       * categorical columns
       * at least one messy value (inconsistent capitalization, missing values, mixed types)
       * one column that is the *target* variable
 - Keep domain-appropriate columns.

3. model_steps (5–12 steps)  
 - MUST be a JSON list describing an ML pipeline, such as:
       * "Split data into train and test sets (80/20)"
       * "One-hot encode categorical variables"
       * "Standardize numeric features"
       * "Train RandomForestClassifier"
       * "Perform grid search over max_depth"
       * "Evaluate accuracy and F1 score"
       * "Compute confusion matrix"
       * "Generate test-set predictions"
 - Steps MUST be written in clear English.

4. model_results  
 - MUST be a JSON dictionary.  
 - MUST include realistic metrics based on the modeling type:  
       * For classification:
            {"accuracy": ..., "f1": ..., "precision": ..., "recall": ...}
       * For regression:
            {"rmse": ..., "mae": ..., "r2": ...}
 - MAY include:
       * top feature importances
       * confusion matrix
       * predicted vs actual values summary
       * hyperparameter choices

 - Numbers should logically reflect the dataset (but do NOT need to be computed exactly).

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

            if not all(k in example for k in ["purpose", "raw_table", "model_steps", "model_results"]):
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
def generate_synthetic_modeling(n=100, outfile="synthetic_modeling.jsonl"):
    print(f"\n🤖 Generating {n} synthetic modeling examples...\n")

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

    print(f"\n🎉 Done! Saved ML modeling dataset → {outfile}")


# ----------------------------
#  CLI
# ----------------------------
if __name__ == "__main__":
    generate_synthetic_modeling(n=300, outfile="synthetic_modeling_akshita.jsonl")
