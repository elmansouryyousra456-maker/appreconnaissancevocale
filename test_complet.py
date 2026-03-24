from fastapi import FastAPI
from app.api.routes.audio import router as audio_router
from app.api.routes.transcription import router as transcription_router
from app.api.routes.resume import router as resume_router

print("=== Création de l'application ===")
app = FastAPI()

print("Inclusion des routes...")
app.include_router(audio_router, prefix="/api/audio", tags=["Audio"])
app.include_router(transcription_router, prefix="/api/transcription", tags=["Transcription"])
app.include_router(resume_router, prefix="/api/resume", tags=["Résumé"])

@app.get("/routes_test")
def routes_test():
    routes_list = []
    for route in app.routes:
        routes_list.append(f"{route.path} - {route.methods}")
    return {"routes": routes_list}

print("=== Routes enregistrées ===")
for route in app.routes:
    print(f"  {route.path} - {route.methods}")

print("=== Test terminé ===")