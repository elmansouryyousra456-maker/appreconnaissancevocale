import aiofiles
import os
import uuid

async def save_upload_file(upload_file, destination_dir: str):
    """
    Sauvegarde un fichier uploadé et retourne le chemin et la taille
    """
    # Créer un nom de fichier unique
    file_extension = os.path.splitext(upload_file.filename)[1].lower()
    unique_filename = f"audio_{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(destination_dir, unique_filename)
    
    # Sauvegarder le fichier
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    # Obtenir la taille
    file_size = os.path.getsize(file_path)
    
    return {
        "file_path": file_path,
        "file_size": file_size,
        "original_filename": upload_file.filename,
        "saved_filename": unique_filename
    }