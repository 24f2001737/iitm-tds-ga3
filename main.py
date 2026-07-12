import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

# ----------------------------
# Initialize Gemini
# ----------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not found.")

client = genai.Client(api_key=api_key)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Multimodal Image QA API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Model
# ----------------------------
class ImageRequest(BaseModel):
    image_base64: str
    question: str

# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {"status": "running"}

# ----------------------------
# Main Endpoint
# ----------------------------
@app.post("/answer-image")
def answer_image(req: ImageRequest):
    try:

        prompt = (
            "Answer the user's question using ONLY the image.\n"
            "If the answer is numeric, return ONLY the number without "
            "currency symbols, commas, units, or extra words.\n"
            "Return only the final answer."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": req.image_base64,
                    }
                },
                prompt,
                req.question,
            ],
        )

        answer = response.text.strip()

        return {
            "answer": str(answer)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
