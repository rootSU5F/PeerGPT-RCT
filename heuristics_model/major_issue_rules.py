import pandas as pd
import csv
import requests
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ✅ Replace with your actual DeepSeek API key
API_KEY = "sk-34e42024cf2143eea141f61ec7cfbed4"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ✅ Static rules definition (only included once per prompt)
BIAS_RULES = """
You are a clinical research methodology expert.

Analyze the following RCT abstract and determine whether it shows signs of any of the following 5 major biases, based on the strict criteria below.

=======================
BIAS HEURISTIC DEFINITIONS:

1. Sample Size Bias (sample_size_detected):
- If sample size < 500: "This was a small study, which increases the possibility that results are due to chance alone"
- If sample size is 500 to 1000: "Relatively small study, limits ability to detect small effect size"

2. Primary Outcome Events Bias (primary_outcome_events_detected):
- If < 30 events in either group: "Low event rate increases chance findings and wide confidence intervals"

3. Composite Outcome Bias (composite_outcome_detected):
- Composite primary outcome is used: "Harder to interpret and driven by least clinically important component"
- Look for multiple endpoints joined by 'and/or' or terms like 'composite', 'combined'

4. Placebo Bias (placebo_detected):
- Drug vs. Standard of Care: Consider as lacking placebo
- No mention of placebo: Introduces potential for confounding and adjudication bias

5. Blinding Bias (blinding_detected):
- If blinding is not provided or not mentioned: Consider as lack of blinding
=======================

Return your results in this exact CSV format:
abstract_id,abstract,timestamp,placebo_detected,blinding_detected,sample_size_detected,composite_outcome_detected,primary_outcome_events_detected,total_biases,has_any_bias

Rules:
- Use "yes" or "no" for each *_detected column.
- total_biases = count of "yes"
- has_any_bias = "yes" if total_biases > 0 else "no"
- Return only one line of CSV output.

Abstract:
""".strip()

# 🔁 Send to DeepSeek API
def generate_response(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
    }
    while True:
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=45)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("⚠️ Rate limited. Sleeping 5 seconds...")
                time.sleep(5)
            else:
                raise e
        except Exception as e:
            print(f"❌ Request failed: {e}")
            raise

# 👷‍♂️ Process one abstract
def process_abstract(i, text):
    if not isinstance(text, str) or not text.strip():
        return None
    prompt = f"{BIAS_RULES}\n{text.strip()}"
    try:
        print(f"⏳ [Abstract {i+1}] Sending request...")
        result = generate_response(prompt).strip()
        csv_buffer = io.StringIO(result)
        parsed_row = next(csv.reader(csv_buffer, skipinitialspace=True))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [f"A{i+1:04d}", text.strip(), timestamp] + parsed_row[3:]  # Skip ID, abstract, timestamp in model response
    except Exception as e:
        print(f"❌ [Abstract {i+1}] Error: {e}")
        return None

# ✅ Load CSV abstracts
df = pd.read_csv("docs/pubmed_abstracts.csv")
if 'Abstract' not in df.columns:
    raise ValueError("CSV must contain a column named 'Abstract'")

# ✅ Optional: test on subset
# df = df.head(20)

# ✅ Parallel execution
results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {
        executor.submit(process_abstract, i, abstract): i
        for i, abstract in enumerate(df['Abstract'])
    }

    for future in as_completed(futures):
        result = future.result()
        if result:
            results.append(result)

# ✅ Save output CSV
with open("bias_results_parallel.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "abstract_id", "abstract", "timestamp",
        "placebo_detected", "blinding_detected", "sample_size_detected",
        "composite_outcome_detected", "primary_outcome_events_detected",
        "total_biases", "has_any_bias"
    ])
    writer.writerows(results)

print("✅ All done! Results saved to bias_results_parallel.csv")
