import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

app = FastAPI(title="Milo Prime API")

# Configuration CORS pour que ton appli communique avec Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Récupération de la clé API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("La variable d'environnement GEMINI_API_KEY est manquante !")

genai.configure(api_key=GEMINI_API_KEY)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    mode: str  # 'perso' ou 'fiverr'
    history: List[ChatMessage]

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Instructions selon le mode choisi sur ton interface
        if req.mode == "fiverr":
            system_instruction = (
                "Tu es Milo, l'assistant commercial pro de Joffranck. "
                "Rédige une réponse professionnelle, claire, polie et engageante en anglais "
                "pour un client Fiverr. Sois direct et vends tes compétences de monteur."
            )
        else:
            system_instruction = (
                "Tu es Milo, le copilote IA et pote de Joffranck. "
                "Tu l'aides pour ses montages de Shorts vidéo, ses idées de B-roll, et ses tendances. "
                "Parle comme un monteur cool, dynamique, utilise un langage familier de pote (wesh, reuf, gros, charbonner). "
                "Sois ultra motivant et va droit au but."
            )

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction
        )

        # Reconstruction de l'historique
        contents = []
        for msg in req.history:
            contents.append({"role": msg.role, "parts": [msg.content]})
        contents.append({"role": "user", "parts": [req.message]})

        response = model.generate_content(contents)
        return {"reply": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Milo Prime Server Live ⚡"}
