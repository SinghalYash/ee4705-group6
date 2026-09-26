from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_model():

    print("Loading Qwen3-VL-8B...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
        )
    )

    print("Qwen3-VL-8B loaded successfully.")

    return model, processor


def ask_qwen(
    model,
    processor,
    image,
    query,
    max_new_tokens=200,
):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": query,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(
        model.device
    )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output[0]

def qwen3_bbox_to_pixels(
    bbox,
    image_width,
    image_height,
):
    """
    Convert Qwen3's normalized 0-1000 bbox:

    [x_min, y_min, x_max, y_max]

    into image pixel coordinates.
    """

    x_min, y_min, x_max, y_max = bbox

    return [
        round(x_min / 1000 * image_width),
        round(y_min / 1000 * image_height),
        round(x_max / 1000 * image_width),
        round(y_max / 1000 * image_height),
    ]