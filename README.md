# parliament-voting-records

- Pythorch
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

- Otros OCR
  pip install pytesseract
  sudo apt install tesseract-ocr tesseract-ocr-spa

- Orden de notebooks y data

* 0. dataset_etiquetado_zonas -> Etiquetado de zonas de encabezado, filas y pie
* 1. scrapping-> traer nuevos documentos
* 2. classiffier -> usa el modelo de resnet para classificar los documentos escaneados y agrega los nuevos a procesamiento_todas_votaciones
* 3. procesamiento_todas_votaciones: procesa todas las imagenes tanto las anteriores como las nuevas
