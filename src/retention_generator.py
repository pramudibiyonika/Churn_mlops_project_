from openai import OpenAI
import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_retention_offer(customer):

    prompt = f"""
    A telecom customer is likely to churn.

    Customer information:
    - Internet Service: {customer.get("InternetService")}
    - Tenure: {customer.get("tenure")} months
    - Contract Type: {customer.get("Contract")}
    - Monthly Charges: {customer.get("MonthlyCharges")}

    Generate a short personalized retention offer
    to convince the customer to stay.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content