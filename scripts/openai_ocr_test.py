from utils_openai_ocr import process_image_ocr

result = process_image_ocr(
    image_path="./test/zona_encabezado.jpg",
    resize_percent=90,
    model="gpt-5-mini",
    max_tokens=2500,
    prompt="En base a la imagen extrae un json con el tipo (Asistencia o Votación), fecha, hora, presidente y asunto. Solo quiero el json sin comentarios adicionales. Si no encuentras alguno de esos campos pon null",
    output_path="resultado_encabezado.json",
)


result = process_image_ocr(
    image_path="./test/zona_voto_oral_1.jpg",
    resize_percent=90,
    model="gpt-5-mini",
    max_tokens=2500,
    prompt='Extrae de la imagen la lista de votos orales y devuelve solo un JSON con la clave votos_orales como arreglo de objetos con el formato {"nombre": NOMBRE_EN_MAYÚSCULAS, "voto": "A favor"|"En contra"|"Abstención"}; si falta algún dato usa null; solo quiero el json, no agregues texto adicional ni comentarios.',
    output_path="resultado_pie_1.json",
)


result = process_image_ocr(
    image_path="./test/zona_voto_oral_2.jpg",
    resize_percent=90,
    model="gpt-5-mini",
    max_tokens=2500,
    prompt='Extrae de la imagen la lista de votos orales y devuelve solo un JSON con la clave votos_orales como arreglo de objetos con el formato {"nombre": NOMBRE_EN_MAYÚSCULAS, "voto": "A favor"|"En contra"|"Abstención"}; si falta algún dato usa null; solo quiero el json, no agregues texto adicional ni comentarios.',
    output_path="resultado_pie_2.json",
)


result = process_image_ocr(
    image_path="./test/zona_voto_oral_3.jpg",
    resize_percent=90,
    model="gpt-5-mini",
    max_tokens=2500,
    prompt='Extrae de la imagen la lista de votos orales y devuelve solo un JSON con la clave votos_orales como arreglo de objetos con el formato {"nombre": NOMBRE_EN_MAYÚSCULAS, "voto": "A favor"|"En contra"|"Abstención"}; si falta algún dato usa null; solo quiero el json, no agregues texto adicional ni comentarios.',
    output_path="resultado_pie_3.json",
)


result = process_image_ocr(
    image_path="./test/zona_voto_oral_4.jpg",
    resize_percent=90,
    model="gpt-5-mini",
    max_tokens=2500,
    prompt='Extrae de la imagen la lista de votos orales y devuelve solo un JSON con la clave votos_orales como arreglo de objetos con el formato {"nombre": NOMBRE_EN_MAYÚSCULAS, "voto": "A favor"|"En contra"|"Abstención"}; si falta algún dato usa null; solo quiero el json, no agregues texto adicional ni comentarios.',
    output_path="resultado_pie_4.json",
)
