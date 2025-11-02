import base64
import json
import os
import re
from io import BytesIO

import openai
from dotenv import load_dotenv
from PIL import Image

# Cargar .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Inicializar cliente OpenAI
client = openai.OpenAI(api_key=api_key)


# 🧹 Limpiar y parsear JSON de respuestas de modelos
def parse_json_response(content):
    """
    Intenta parsear JSON de forma robusta, manejando casos donde viene envuelto
    en bloques de código markdown o con comillas adicionales.

    Args:
        content: String que posiblemente contiene JSON

    Returns:
        dict o el contenido original si no se pudo parsear
    """
    # Guardar el contenido original por si falla todo
    original_content = content

    try:
        # 1. Eliminar bloques de código markdown con diferentes variantes
        # Patrones: ```json, ```JSON, ``` al inicio, ``` al final
        content = re.sub(r"^```(?:json|JSON)?\s*\n?", "", content.strip())
        content = re.sub(r"\n?```\s*$", "", content.strip())

        # 2. Eliminar comillas triples al inicio y final (''' o """)
        content = re.sub(r"^[\'\"\`]{3,}\s*", "", content.strip())
        content = re.sub(r"\s*[\'\"\`]{3,}$", "", content.strip())

        # 3. Eliminar posibles etiquetas "json" sueltas
        content = re.sub(r"^json\s*\n?", "", content.strip(), flags=re.IGNORECASE)

        # 4. Limpiar espacios en blanco adicionales
        content = content.strip()

        # 5. Intentar parsear como JSON
        parsed = json.loads(content)
        return parsed

    except (json.JSONDecodeError, Exception) as e:
        # Si falla, intentar con el contenido original sin limpiar
        try:
            return json.loads(original_content)
        except:
            # Si todo falla, retornar el contenido original
            return original_content


# 🔧 Redimensionar en memoria
def resize_image_in_memory(image_path, resize_percent=50):
    img = Image.open(image_path)
    original_size = img.size

    new_width = int(original_size[0] * resize_percent / 100)
    new_height = int(original_size[1] * resize_percent / 100)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Convertir a bytes en memoria
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    print(f"📏 Imagen original: {original_size}, redimensionada a: ({new_width}, {new_height})")
    return buffer


# 🧪 Codificar imagen desde buffer
def encode_image_base64_from_buffer(buffer):
    return base64.b64encode(buffer.read()).decode("utf-8")


# 🧠 Extraer texto con modelo seleccionable
def extract_text_from_image(
    base64_image,
    model,
    max_tokens,
    prompt,
):
    # Modelos disponibles con visión y sus precios por 1K tokens
    vision_models = {
        "gpt-5": {"input": 0.00125, "output": 0.010, "use_max_completion_tokens": True},
        "gpt-5-mini": {"input": 0.00025, "output": 0.002, "use_max_completion_tokens": True},
    }

    # Validación modelo
    if model not in vision_models:
        print(f"⚠️ Modelo {model} no soporta visión. Usando gpt-4o")
        model = "gpt-4o"

    pricing = vision_models[model]
    print(f"🤖 Usando modelo: {model}")

    # Construcción de request compatible con todos los modelos
    request_params = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
    }

    # ⚠️ Diferencia clave: GPT-5 usa max_completion_tokens
    if pricing["use_max_completion_tokens"]:
        request_params["max_completion_tokens"] = max_tokens
    else:
        request_params["max_tokens"] = max_tokens

    # Llamada a la API
    response = client.chat.completions.create(**request_params)

    # Cálculo de costos
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = prompt_tokens + completion_tokens

    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000
    print(
        f"\n📊 Tokens usados: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
    )
    print(f"💵 Costo real estimado: ${cost:.5f} USD")

    # Retornar contenido y metadata
    return {
        "content": response.choices[0].message.content,
        "meta": {
            "model": model,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "cost_usd": round(cost, 5),
            "pricing": {"input_per_1k": pricing["input"], "output_per_1k": pricing["output"]},
        },
    }


# 🚀 Función principal para probar diferentes configuraciones
def process_image_ocr(
    image_path,
    resize_percent=40,
    model="gpt-5-mini",
    max_tokens=2000,
    prompt="En base a la imagen extrae un json con la fecha, hora, presidente y asunto. Solo quiero el json sin comentarios adicionales.",
    output_path=None,
):
    """
    Procesa una imagen con OCR usando diferentes modelos y configuraciones

    Args:
        image_path: Ruta a la imagen
        resize_percent: Porcentaje de redimensionado (100 = tamaño original)
        model: Modelo de OpenAI a usar ('gpt-4o', 'gpt-4o-mini', etc.)
        max_tokens: Máximo de tokens en la respuesta
        prompt: Prompt personalizado para el OCR
        output_path: Ruta donde guardar el JSON de salida (opcional)

    Returns:
        dict: Diccionario con estructura {"output": contenido_ocr, "meta": metadata}
    """
    # Redimensionar en memoria
    img_buffer = resize_image_in_memory(image_path, resize_percent)

    # Codificar a base64
    b64_img = encode_image_base64_from_buffer(img_buffer)

    # Extraer texto con el modelo seleccionado
    result = extract_text_from_image(b64_img, model, max_tokens, prompt)

    # Intentar parsear el contenido si es JSON válido (con limpieza robusta)
    content = result["content"]
    parsed_content = parse_json_response(content)

    # Verificar si se parseó correctamente
    if isinstance(parsed_content, (dict, list)):
        print("✅ Contenido parseado como JSON exitosamente")
    else:
        print("ℹ️ Contenido mantenido como texto (no es JSON válido)")

    # Construir la salida con la estructura solicitada
    output_json = {"output": parsed_content, "meta": result["meta"]}

    # Guardar en archivo si se especificó output_path
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON guardado en: {output_path}")

    return output_json


# ========================================
# 📚 EJEMPLO DE USO
# ========================================
#
# Este módulo está diseñado para ser importado en otros archivos Python.
# Para usarlo, importa la función principal process_image_ocr:
#
# ```python
# from utils_openai_ocr import process_image_ocr
#
# # Ejemplo con todos los parámetros
# result = process_image_ocr(
#     image_path="ruta/a/tu/imagen.jpg",           # Ruta a la imagen (requerido)
#     resize_percent=80,                            # Porcentaje de redimensionado (default: 40)
#     model="gpt-5-mini",                          # Modelo a usar: "gpt-5" o "gpt-5-mini" (default: "gpt-5-mini")
#     max_tokens=2000,                             # Máximo de tokens en la respuesta (default: 2000)
#     prompt="Tu prompt personalizado aquí",       # Prompt para el OCR (default: extrae fecha, hora, presidente y asunto)
#     output_path="resultado.json"                 # Ruta para guardar el JSON (opcional, default: None)
# )
#
# # Acceder a los resultados
# print(result["output"])                          # Contenido del OCR (parseado como dict si es JSON)
# print(result["meta"]["tokens"]["total"])         # Total de tokens usados
# print(result["meta"]["tokens"]["prompt"])        # Tokens del prompt
# print(result["meta"]["tokens"]["completion"])    # Tokens de la respuesta
# print(result["meta"]["cost_usd"])                # Costo en dólares
# print(result["meta"]["model"])                   # Modelo utilizado
# print(result["meta"]["pricing"])                 # Precios por 1K tokens
# ```
#
# ========================================
