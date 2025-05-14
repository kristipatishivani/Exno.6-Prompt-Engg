# Exno.6-Prompt-Engg
# Date:
# Register no.
# Aim: Development of Python Code Compatible with Multiple AI Tools



# Algorithm: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.
Objective: Automate interaction with multiple AI tools → Collect outputs → Compare → Derive insights Tools Integrated:

OpenAI (GPT-4 or GPT-3.5)

Cohere (Command R or similar)

Hugging Face (via Inference API)

**Python Code: Multi-AI Output Comparison**

import openai
import cohere
import requests
import difflib
from transformers import pipeline
from textblob import TextBlob

# Step 1: Setup API keys (insert yours here)
openai.api_key = "YOUR_OPENAI_API_KEY"
co = cohere.Client("YOUR_COHERE_API_KEY")
HF_API_TOKEN = "YOUR_HUGGINGFACE_API_KEY"

# Step 2: Unified prompt
prompt = "Suggest three healthy lifestyle tips for managing stress."

# Step 3: Fetch response from OpenAI
```
def get_openai_response(prompt):
        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()
```
# Step 4: Fetch response from Cohere
```
def get_cohere_response(prompt):
        response = co.generate(
        model='command-r',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.generations[0].text.strip()
```
# Step 5: Fetch response from Hugging Face (e.g., google/flan-t5-base)
```
def get_huggingface_response(prompt):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    response = requests.post(api_url, headers=headers, json={"inputs": prompt})
    return response.json()[0]['generated_text'].strip()
```
# Step 6: Compare responses using difflib
```
def compare_responses(r1, r2, r3):
    similarity_1_2 = difflib.SequenceMatcher(None, r1, r2).ratio()
    similarity_2_3 = difflib.SequenceMatcher(None, r2, r3).ratio()
    similarity_1_3 = difflib.SequenceMatcher(None, r1, r3).ratio()
    return round((similarity_1_2 + similarity_2_3 + similarity_1_3) / 3, 2)
```
# Step 7: Analyze sentiment
```
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
```
# Step 8: Execute everything
```
def run_analysis(prompt):
    print("🧠 Prompt:", prompt)
    
    oai = get_openai_response(prompt)
    coh = get_cohere_response(prompt)
    hf = get_huggingface_response(prompt)

    print("\n🔹 OpenAI:\n", oai)
    print("\n🔹 Cohere:\n", coh)
    print("\n🔹 Hugging Face:\n", hf)

    similarity = compare_responses(oai, coh, hf)
    sentiment_avg = round((analyze_sentiment(oai) + analyze_sentiment(coh) + analyze_sentiment(hf)) / 3, 2)

    print(f"\n📊 Average Similarity: {similarity}")
    print(f"📈 Average Sentiment Score: {sentiment_avg}")

    insight = "✅ Responses are consistent" if similarity > 0.6 else "⚠️ High variance in outputs"
    sentiment_note = "😊 Positive tone overall" if sentiment_avg > 0 else "😐 Neutral or negative tone"
    
    print(f"\n💡 Insight: {insight}, {sentiment_note}")

# Run the system
run_analysis(prompt)

# Result: The corresponding Prompt is executed successfully
The prompt is executed successfully.

Outputs from OpenAI, Cohere, and Hugging Face are printed.

Similarity score and sentiment analysis provide automated insights on alignment and tone.
