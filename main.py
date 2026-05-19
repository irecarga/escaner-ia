import os
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from paddleocr import PaddleOCR

app = FastAPI(title="Escáner IA - OCR API")

# Permitir conexiones desde cualquier origen (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el motor de PaddleOCR en español
# download_idx=True descarga los modelos necesarios automáticamente
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='es', download_idx=True)
except Exception as e:
    print(f"Error al inicializar PaddleOCR: {e}")
    ocr = None

@app.get("/")
def read_root():
    return {"status": "online", "message": "Servidor de Escáner IA activo"}

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    if ocr is None:
        raise HTTPException(status_code=500, detail="El motor OCR no está disponible.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")

    try:
        # Leer la imagen enviada
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Convertir la imagen a un formato compatible con PaddleOCR (numpy array)
        img_np = np.array(image)
        
        # Ejecutar el reconocimiento de texto
        result = ocr.ocr(img_np, cls=True)
        
        # Extraer solo las líneas de texto detectadas
        detected_text = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0] # Texto detectado
                confidence = line[1][1] # Nivel de confianza del OCR
                detected_text.append(text)
                
        return {
            "success": True,
            "lines": detected_text,
            "raw_text": " ".join(detected_text)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
