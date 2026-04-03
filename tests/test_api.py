from pathlib import Path


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"].startswith("Backend AssistEduc")


def test_audio_upload_and_crud(client, wav_bytes):
    upload_response = client.post(
        "/api/audio/upload",
        files={"file": ("lesson.wav", wav_bytes, "audio/wav")},
    )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["filename"] == "lesson.wav"
    assert payload["status"] == "uploaded"
    assert payload["duration"] is not None

    audio_id = payload["id"]
    stored_path = Path(payload["file_path"])
    assert stored_path.exists()

    list_response = client.get("/api/audio")
    assert list_response.status_code == 200
    assert any(item["id"] == audio_id for item in list_response.json())

    patch_response = client.patch(
        f"/api/audio/{audio_id}",
        json={"filename": "lesson-final.wav", "status": "processed"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["filename"] == "lesson-final.wav"
    assert patch_response.json()["status"] == "processed"

    delete_response = client.delete(f"/api/audio/{audio_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["message"]
    assert not stored_path.exists()


def test_transcription_and_summary_flow(client, uploaded_audio):
    audio_id = uploaded_audio["id"]

    transcription_response = client.post(
        "/api/transcription/transcribe",
        json={"audio_id": audio_id, "language": "fr"},
    )
    assert transcription_response.status_code == 200, transcription_response.text
    transcription_payload = transcription_response.json()
    assert transcription_payload["audio_id"] == audio_id
    assert transcription_payload["segments"]

    transcription_id = transcription_payload["id"]

    summary_response = client.post(
        "/api/resume/generate",
        json={"transcription_id": transcription_id, "ratio": 0.4},
    )
    assert summary_response.status_code == 200, summary_response.text
    summary_payload = summary_response.json()
    assert summary_payload["transcription_id"] == transcription_id
    assert summary_payload["summary"]
    assert summary_payload["stats"]["summary_length"] > 0

    latest_response = client.get(f"/api/resume/{transcription_id}")
    assert latest_response.status_code == 200
    latest_payload = latest_response.json()
    assert latest_payload["transcription_id"] == transcription_id

    summary_id = latest_payload["id"]
    update_response = client.patch(
        f"/api/resume/item/{summary_id}",
        json={"summary": "Resume modifie", "method": "manual"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["summary"] == "Resume modifie"
    assert update_response.json()["method"] == "manual"


def test_validation_errors(client):
    invalid_upload = client.post(
        "/api/audio/upload",
        files={"file": ("notes.txt", b"bonjour", "text/plain")},
    )
    assert invalid_upload.status_code == 400

    invalid_summary = client.post(
        "/api/resume/generate",
        json={"transcription_id": "missing", "ratio": 1.5},
    )
    assert invalid_summary.status_code == 422
