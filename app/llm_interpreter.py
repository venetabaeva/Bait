import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def interpret_query(user_input, factors):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    Use GPT to map a user query to factor filters.
    Expected output: {{"Persona": "Sponsor", "Condition": "High risk"}}

    You are a BA Advisor Agent. The possible factors are: {', '.join(factors)}.
    User query: "{user_input}"
    Identify which factor values this query is referring to.
    Respond with a valid JSON object (e.g., {{"Persona": "Sponsor"}}).
    If unsure, respond with {{}}.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200
        )
        result_text = response.choices[0].message.content.strip()

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON returned: {result_text}"}

    except Exception as e:
        return {"error": str(e)}
