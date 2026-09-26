import os
import base64
from io import BytesIO

from openai import OpenAI


client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL = "qwen3-vl-flash"

def image_to_base64(image):
    """
    Convert a PIL image into a Base64 data URL
    that can be sent to Qwen3-VL.
    """

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def ask_qwen(image, prompt):
    """
    Send an image + prompt to Qwen3-VL Flash through
    the DashScope cloud API.

    model and processor are kept as arguments so that
    the rest of the perception code does not need to change.
    """

    image_data = image_to_base64(image)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


def bbox_to_pixels(bbox, image_width, image_height):
    """
    Convert Qwen normalized 0-1000 coordinates
    into pixel coordinates.
    """

    x_min, y_min, x_max, y_max = bbox

    return [
        int(x_min / 1000 * image_width),
        int(y_min / 1000 * image_height),
        int(x_max / 1000 * image_width),
        int(y_max / 1000 * image_height),
    ]