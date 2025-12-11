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
#  Synthetic Statistical Testing Prompt
# ----------------------------
GEN_PROMPT = """
You are an expert in statistics, hypothesis testing, and statistical workflows.

Your job is to generate a NEW synthetic EXAMPLE in the [DOMAIN] domain that imitates
a statistical testing dataset.

You MUST return a JSON object with EXACTLY these fields:

{
    "purpose": "...",
    "raw_table": "...CSV string...",
    "stat_test_steps": [... list of statistical testing steps ...],
    "stat_test_results": {... JSON dictionary of test outputs ...}
}

-------------------------------------------
RULES FOR GENERATING THE EXAMPLE:
-------------------------------------------

1. PURPOSE
 - 1–2 sentences describing the statistical question.
 - Examples:
     * "Determine whether average wait times differ between departments."
     * "Test if purchase conversion rates vary across marketing channels."
     * "Evaluate whether temperature anomalies differ between years."

2. RAW TABLE
 - MUST be a CSV string.
 - MUST contain 8–15 rows.
 - MUST include:
       * at least one numeric column
       * at least one categorical column
       * some messy or inconsistent values
       * domain-appropriate fields
 - One column MUST be relevant to the hypothesis test.

3. stat_test_steps (5–10 steps)
 - MUST be a JSON list of statistical analysis operations, such as:
       * "Form null and alternative hypotheses"
       * "Check normality using Shapiro-Wilk test"
       * "Split data into two groups based on Category"
       * "Conduct two-sample t-test"
       * "Compute effect size (Cohen's d)"
       * "Perform chi-square test of independence"
       * "Compare variances using Levene's test"
       * "Compute correlation coefficient"
 - Steps MUST be readable English (not code).

4. stat_test_results
 - MUST be a JSON dictionary containing:
       * the test performed
       * test statistic
       * p-value
       * decision: "reject null" or "fail to reject null"
       * any secondary outputs (effect size, confidence intervals, etc.)
 - Values MUST be realistic (but do not need exact numerical correctness).
 - Example:
       {
         "test": "independent t-test",
         "t_stat": -1.83,
         "p_value": 0.078,
         "decision": "fail to reject null",
         "effect_size": 0.45
       }

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

            if not all(k in example for k in ["purpose", "raw_table", "stat_test_steps", "stat_test_results"]):
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
def generate_synthetic_stat_tests(n=100, outfile="synthetic_stat_tests.jsonl"):
    print(f"\n📈 Generating {n} synthetic statistical testing examples...\n")

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

    print(f"\n🎉 Done! Saved synthetic statistical testing dataset → {outfile}")


# ----------------------------
#  CLI
# ----------------------------
if __name__ == "__main__":
    generate_synthetic_stat_tests(n=300, outfile="synthetic_stat_tests_akshita.jsonl")
