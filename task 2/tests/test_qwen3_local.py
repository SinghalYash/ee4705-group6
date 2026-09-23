from qwen3_local import load_model


model, processor = load_model()


print(
    "\nModel device:",
    next(model.parameters()).device
)