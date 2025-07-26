import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def interpret_query(user_input, factors):
    """
    Use GPT to map a user query to factor filters.
    Example output: {"Persona": "Sponsor", "Condition": "High risk"}
    """
    prompt = f"""
    You are a BA Advisor Agent. The possible factors are: {', '.join(factors)}.
    User query: "{user_input}".
    Identify which factor values this query is referring to.
    Respond with a JSON dictionary, e.g. {{"Persona": "Sponsor"}}.
    If unsure, return an empty JSON object.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        result_text = response.choices[0].message.content.strip()
        return json.loads(result_text)
    except Exception as e:
        return {"error": str(e)}
