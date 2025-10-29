# parliament-voting-records

- Pythorch
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

- PdfToImage
  sudo apt update && sudo apt install poppler-utils -y

- Otros OCR
  pip install pytesseract easyocr  
  pip install 'python-doctr[torch]'  
  pip install paddleocr==2.10.0  
  pip install paddlepaddle==2.6.2  
  sudo apt install tesseract-ocr tesseract-ocr-spa

- Orden de notebooks y data

* 0. dataset_etiquetado_zonas -> Etiquetado de zonas de encabezado, filas y pie
* 1. scrapping-> traer nuevos documentos
* 2. classiffier -> usa el modelo de resnet para classificar los documentos escaneados y agrega los nuevos a procesamiento_todas_votaciones
* 3. procesamiento_todas_votaciones: procesa todas las imagenes tanto las anteriores como las nuevas
