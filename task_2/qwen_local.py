import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():
    """
    Load Qwen2.5-VL and its processor.

    Returns
    -------
    model:
        The Qwen vision-language model.

    processor:
        Converts images/text into the numerical format
        required by Qwen, and converts generated tokens
        back into text.
    """

    print("Loading Qwen2.5-VL...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
    )

    print("Model loaded successfully.")

    return model, processor


# ==========================================================
# ASK QWEN
# ==========================================================

def ask_qwen(
    model,
    processor,
    image,
    query,
    max_new_tokens=200,
):
    """
    Give Qwen an image and a natural-language query.

    Parameters
    ----------
    model:
        Loaded Qwen model.

    processor:
        Loaded Qwen processor.

    image:
        PIL image to analyse.

    query:
        Natural-language question about the image.

    max_new_tokens:
        Maximum length of Qwen's generated response.

    Returns
    -------
    str:
        Qwen's text response.
    """

    # ------------------------------------------------------
    # Create multimodal conversation
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Apply Qwen's chat format
    # ------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ------------------------------------------------------
    # Extract image information
    # ------------------------------------------------------

    image_inputs, video_inputs = (
        process_vision_info(messages)
    )

    # ------------------------------------------------------
    # Convert image + text into tensors
    # ------------------------------------------------------

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Put inputs on the same device as the model.
    inputs = inputs.to(model.device)

    # ------------------------------------------------------
    # Run Qwen inference
    # ------------------------------------------------------

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    # ------------------------------------------------------
    # Remove the original prompt tokens
    # ------------------------------------------------------

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    # ------------------------------------------------------
    # Convert generated tokens back into text
    # ------------------------------------------------------

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]