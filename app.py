import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "handwritten_digit_model.keras"
)

model = tf.keras.models.load_model(model_path)

def predict_digit(image):
    if image is None:
        return "Please draw or upload a digit."

    image = Image.fromarray(np.uint8(image)).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28))

    image = np.array(image).astype("float32") / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image, verbose=0)[0]
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    return f"Predicted Digit: {digit} | Confidence: {confidence:.2f}%"

with gr.Blocks(title="AI Handwritten Digit Recognition") as demo:
    gr.Markdown("# AI Handwritten Digit Recognition")
    gr.Markdown("Draw or upload a digit from 0 to 9.")

    drawing = gr.Sketchpad(
        type="numpy",
        image_mode="L",
        canvas_size=(280, 280)
    )

    upload = gr.Image(type="numpy", image_mode="RGB")

    result = gr.Textbox(label="Prediction")
    predict = gr.Button("Predict Digit", variant="primary")
    clear = gr.ClearButton([drawing, upload, result])

    def run_prediction(draw_image, uploaded_image):
        selected = draw_image if draw_image is not None else uploaded_image
        return predict_digit(selected)

    predict.click(
        fn=run_prediction,
        inputs=[drawing, upload],
        outputs=result
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
