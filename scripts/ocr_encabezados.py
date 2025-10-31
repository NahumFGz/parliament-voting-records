from utils_openai_ocr import process_image_ocr

result = process_image_ocr(
    image_path="./zona_encabezado.jpg",
    resize_percent=90,
    model="gpt-5-mini",
    max_tokens=2500,
    prompt="En base a la imagen extrae un json con la fecha, hora, presidente y asunto. Solo quiero el json sin comentarios adicionales. Si no encuentras alguno de esos campos pon None",
    output_path="resultado_encabezado.json",
)
