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

        # 5. Reemplazar valores Python por valores JSON válidos
        # Importante: usar word boundaries para evitar reemplazos incorrectos
        content = re.sub(r"\bNone\b", "null", content)
        content = re.sub(r"\bTrue\b", "true", content)
        content = re.sub(r"\bFalse\b", "false", content)

        # 6. Intentar decodificar caracteres escapados si viene como string literal
        # Por ejemplo: "{\n  \"key\": \"value\"\n}" -> real JSON
        try:
            # Intentar decodificar con codecs si detectamos escapes
            if "\\n" in content or "\\t" in content or '\\"' in content:
                # Usar decode de unicode-escape para strings con escapes
                content = content.encode().decode("unicode-escape")
        except:
            pass  # Si falla, continuar con el contenido sin decodificar

        # 7. Validar que el JSON esté completo (verificar corchetes/llaves)
        # Contar apertura y cierre de llaves y corchetes
        open_braces = content.count("{")
        close_braces = content.count("}")
        open_brackets = content.count("[")
        close_brackets = content.count("]")

        if open_braces != close_braces or open_brackets != close_brackets:
            print(
                f"⚠️ Advertencia: JSON posiblemente incompleto (llaves: {open_braces}/{close_braces}, corchetes: {open_brackets}/{close_brackets})"
            )

        # 8. Intentar parsear como JSON
        parsed = json.loads(content)
        return parsed

    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ Error parseando JSON: {str(e)[:100]}")
        # Si falla, intentar estrategias adicionales
        try:
            # Estrategia 1: Intentar con el contenido original sin limpiar
            return json.loads(original_content)
        except:
            try:
                # Estrategia 2: Aplicar solo reemplazo de None/True/False al original
                cleaned = re.sub(r"\bNone\b", "null", original_content)
                cleaned = re.sub(r"\bTrue\b", "true", cleaned)
                cleaned = re.sub(r"\bFalse\b", "false", cleaned)
                return json.loads(cleaned)
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
    system_prompt=None,
):
    # Modelos disponibles con visión y sus precios por 1K tokens
    vision_models = {
        "gpt-4o": {"input": 0.0025, "output": 0.010, "use_max_completion_tokens": False},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "use_max_completion_tokens": False},
        "gpt-5": {"input": 0.00125, "output": 0.010, "use_max_completion_tokens": True},
        "gpt-5-mini": {"input": 0.00025, "output": 0.002, "use_max_completion_tokens": True},
    }

    # Validación modelo
    if model not in vision_models:
        print(f"⚠️ Modelo {model} no reconocido. Usando gpt-4o-mini")
        model = "gpt-4o-mini"

    pricing = vision_models[model]
    print(f"🤖 Usando modelo: {model} (JSON mode activado)")

    # Construcción de mensajes
    messages = []

    # Agregar mensaje del sistema si se proporciona
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Agregar mensaje del usuario con la imagen
    messages.append(
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
    )

    # Construcción de request compatible con todos los modelos
    request_params = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},  # ✨ Forzar respuesta en JSON válido
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
    model="gpt-4o-mini",
    max_tokens=2000,
    output_path=None,
    prompt=None,
    system_prompt=None,
):
    """
    Procesa una imagen con OCR usando diferentes modelos y configuraciones

    ⚠️ IMPORTANTE: Esta función fuerza la respuesta en JSON válido usando response_format.
    El prompt DEBE mencionar que la respuesta debe ser en formato JSON, sino el modelo fallará.

    Args:
        image_path: Ruta a la imagen
        resize_percent: Porcentaje de redimensionado (100 = tamaño original)
        model: Modelo de OpenAI a usar ('gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-mini')
        max_tokens: Máximo de tokens en la respuesta
        prompt: Prompt personalizado para el OCR (DEBE mencionar formato JSON)
        output_path: Ruta donde guardar el JSON de salida (opcional)
        system_prompt: Mensaje del sistema para agregar contexto (opcional)

    Returns:
        dict: Diccionario con estructura {"output": contenido_ocr, "meta": metadata}
    """
    # Prompt por defecto si no se proporciona (DEBE mencionar JSON)
    if prompt is None:
        prompt = """Extrae toda la información visible en esta imagen.
        Si contiene tablas, extrae los datos de forma estructurada.
        Responde ÚNICAMENTE en formato JSON válido."""

    # Redimensionar en memoria
    img_buffer = resize_image_in_memory(image_path, resize_percent)

    # Codificar a base64
    b64_img = encode_image_base64_from_buffer(img_buffer)

    # Extraer texto con el modelo seleccionado
    result = extract_text_from_image(b64_img, model, max_tokens, prompt, system_prompt)

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
# Este módulo usa la API de OpenAI con JSON mode activado para garantizar respuestas JSON válidas.
# Para usarlo, importa la función principal process_image_ocr:
#
# ```python
# from utils_openai_ocr import process_image_ocr
#
# # Ejemplo básico (usa prompt por defecto que pide JSON)
# result = process_image_ocr(
#     image_path="ruta/a/tu/imagen.jpg",
#     model="gpt-4o-mini",
# )
#
# # Ejemplo con todos los parámetros
# result = process_image_ocr(
#     image_path="ruta/a/tu/imagen.jpg",           # Ruta a la imagen (requerido)
#     resize_percent=80,                            # Porcentaje de redimensionado (default: 40)
#     model="gpt-4o-mini",                         # Modelo: "gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini" (default: "gpt-4o-mini")
#     max_tokens=2000,                             # Máximo de tokens en la respuesta (default: 2000)
#     prompt="Extrae la tabla y devuelve en JSON", # ⚠️ IMPORTANTE: DEBE mencionar JSON en el prompt
#     system_prompt="Eres un experto en OCR",     # System prompt opcional
#     output_path="resultado.json"                 # Ruta para guardar el JSON (opcional)
# )
#
# # Acceder a los resultados
# print(result["output"])                          # Contenido del OCR (parseado como dict)
# print(result["meta"]["tokens"]["total"])         # Total de tokens usados
# print(result["meta"]["cost_usd"])                # Costo en dólares
# print(result["meta"]["model"])                   # Modelo utilizado
# ```
#
# ✨ Ventajas de JSON mode (response_format):
# - ✅ Garantiza que el modelo SIEMPRE responda con JSON válido
# - ✅ No necesitas parsing complejo ni manejo de markdown
# - ✅ Reduce errores de parseo en ~99%
# - ✅ El modelo falla si no puede generar JSON válido (evita respuestas corruptas)
#
# ⚠️ IMPORTANTE:
# - El prompt DEBE mencionar que quieres JSON (ya incluido en el prompt por defecto)
# - Si usas prompt personalizado, incluye "responde en JSON" o similar
#
# ========================================
