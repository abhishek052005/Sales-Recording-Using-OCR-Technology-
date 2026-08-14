import json
import os

from groq import Groq
from dotenv import load_dotenv

from .models import Invoice


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


SYSTEM_PROMPT = """
You are an invoice data extraction system.

Extract structured invoice information from the OCR text.

Rules:
1. Extract only information present in the OCR text.
2. Never invent or guess information.
3. Use null when information is missing.
4. Extract all identifiable invoice items.
5. Numbers must be returned as numbers.
6. Normalize dates to YYYY-MM-DD when possible.
7. Preserve GSTIN exactly as found.
8. Correct obvious OCR errors only when the intended value is clear.
9. Do not calculate missing values.
10. Return only the requested JSON structure.
"""


def get_invoice_schema():

    return {
        "type": "object",

        "properties": {

            "invoice_number": {
                "type": ["string", "null"]
            },

            "invoice_date": {
                "type": ["string", "null"]
            },

            "vendor": {
                "type": "object",

                "properties": {
                    "name": {
                        "type": ["string", "null"]
                    },
                    "gstin": {
                        "type": ["string", "null"]
                    },
                    "address": {
                        "type": ["string", "null"]
                    },
                    "phone": {
                        "type": ["string", "null"]
                    },
                    "email": {
                        "type": ["string", "null"]
                    }
                },

                "required": [
                    "name",
                    "gstin",
                    "address",
                    "phone",
                    "email"
                ],

                "additionalProperties": False
            },

            "customer": {
                "type": "object",

                "properties": {
                    "name": {
                        "type": ["string", "null"]
                    },
                    "gstin": {
                        "type": ["string", "null"]
                    },
                    "address": {
                        "type": ["string", "null"]
                    },
                    "phone": {
                        "type": ["string", "null"]
                    },
                    "email": {
                        "type": ["string", "null"]
                    }
                },

                "required": [
                    "name",
                    "gstin",
                    "address",
                    "phone",
                    "email"
                ],

                "additionalProperties": False
            },

            "items": {
                "type": "array",

                "items": {
                    "type": "object",

                    "properties": {
                        "description": {
                            "type": "string"
                        },
                        "quantity": {
                            "type": ["number", "null"]
                        },
                        "unit_price": {
                            "type": ["number", "null"]
                        },
                        "tax_rate": {
                            "type": ["number", "null"]
                        },
                        "amount": {
                            "type": ["number", "null"]
                        }
                    },

                    "required": [
                        "description",
                        "quantity",
                        "unit_price",
                        "tax_rate",
                        "amount"
                    ],

                    "additionalProperties": False
                }
            },

            "subtotal": {
                "type": ["number", "null"]
            },

            "tax": {
                "type": ["number", "null"]
            },

            "total": {
                "type": ["number", "null"]
            },

            "currency": {
                "type": ["string", "null"]
            }
        },

        "required": [
            "invoice_number",
            "invoice_date",
            "vendor",
            "customer",
            "items",
            "subtotal",
            "tax",
            "total",
            "currency"
        ],

        "additionalProperties": False
    }


def extract_invoice_data(ocr_text: str):

    if not ocr_text or not ocr_text.strip():
        raise ValueError("OCR text is empty")

    completion = client.chat.completions.create(

        # Your model
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Extract invoice data from this OCR text:

-------------------------
{ocr_text}
-------------------------
"""
            }
        ],

        # IMPORTANT:
        # Do NOT use stream=True for DB extraction
        stream=False,

        # Structured JSON
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_extraction",
                "strict": True,
                "schema": get_invoice_schema()
            }
        },

        temperature=0,

        max_completion_tokens=4096,

        reasoning_effort="medium"
    )

    content = completion.choices[0].message.content

    if not content:
        raise ValueError("Groq returned empty response")

    try:
        data = json.loads(content)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON returned by Groq: {e}"
        )

    # Pydantic validation
    invoice = Invoice.model_validate(data)

    # Return normal Python dict to FastAPI
    return invoice.model_dump()
