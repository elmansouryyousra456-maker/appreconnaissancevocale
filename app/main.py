from fastapi import FastAPI
from app.api.routes.audio import router as audio_router
from app.api.routes.transcription import router as transcription_router
from app.api.routes.resume import router as resume_router

app = FastAPI(title="AssistEduc API", version="1.0.0")

# Inclusion des routes (les préfixes sont déjà dans les routers)
app.include_router(audio_router)
app.include_router(transcription_router)
app.include_router(resume_router)

@app.get("/")
def read_root():
    return {"message": "Backend AssistEduc fonctionne 🚀"}

@app.get("/routes")
def list_routes():
    return [{"path": route.path, "methods": list(route.methods)} for route in app.routes]