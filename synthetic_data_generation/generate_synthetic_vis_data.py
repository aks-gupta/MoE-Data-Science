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
#  Synthetic Visualization Prompt
# ----------------------------
GEN_PROMPT = """
You are an expert in data visualization design, plotting, and exploratory graphics.

Your job is to generate a NEW synthetic EXAMPLE in the [DOMAIN] domain that imitates
a visualization dataset.

You MUST return a JSON object with EXACTLY these fields:

{
    "purpose": "...",
    "raw_table": "...CSV string...",
    "viz_instructions": [... list of human-readable visualization steps ...],
    "viz_spec": {... structured description of the plots ...}
}

-------------------------------------------
RULES FOR GENERATING THE EXAMPLE:
-------------------------------------------

1. PURPOSE
 - 1–2 sentences describing what visualization will help analyze.
 - Example: "Visualize monthly revenue trends and compare categories."

2. RAW TABLE
 - MUST be a CSV string.
 - MUST contain 8–15 rows.
 - MUST include realistic VALUES — some messy:
       * inconsistent capitalization
       * weird category names
       * mixed date formats
       * missing values
       * strings vs numbers mixed
 - Columns must be domain-appropriate.

3. viz_instructions (5–10)
 - MUST be readable English steps describing desired visualizations:
       * "Create a bar chart of sales by category"
       * "Plot monthly trend of temperature"
       * "Visualize distribution of transaction amounts using a histogram"
       * "Compare average wait time by department using a box plot"
 - Each step must be high-level (not code).

4. viz_spec
 - MUST be a JSON dictionary describing one or more plots.
 - Structure must be SIMPLE, like this style:
       {
         "bar_chart": {
             "x": "Category",
             "y": "TotalSales",
             "aggregation": "sum"
         },
         "line_chart": {
             "x": "Month",
             "y": "Revenue",
             "group_by": "Region"
         }
       }
 - No need for real Vega-Lite schema — just clean JSON.
 - Keys must reflect visualization types:
       * "bar_chart"
       * "line_chart"
       * "scatter_plot"
       * "histogram"
       * "box_plot"

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
            example = json.loads(content)

            if not all(k in example for k in ["purpose", "raw_table", "viz_instructions", "viz_spec"]):
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
def generate_synthetic_visualizations(n=100, outfile="synthetic_visualizations.jsonl"):
    print(f"\n📊 Generating {n} synthetic visualization examples...\n")

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

    print(f"\n🎉 Done! Saved → {outfile}")


# ----------------------------
#  CLI
# ----------------------------
if __name__ == "__main__":
    generate_synthetic_visualizations(n=300, outfile="synthetic_viz_akshita.jsonl")
