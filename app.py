
import os
import numpy as np
import gradio as gr
import tensorflow as tf

from PIL import Image, ImageOps

# --------------------------------
# 1. LOAD TRAINED MODEL
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "handwritten_digit_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# --------------------------------
# 2. IMAGE PREPROCESSING
# --------------------------------

def prepare_image(image):

    # Handle Sketchpad dictionary output
    if isinstance(image, dict):
        image = image.get("composite")

    if image is None:
        return None

    image = np.array(image)

    # Convert image to PIL format
    if image.dtype != np.uint8:
        if image.max() <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    # Handle transparent images
    if pil_image.mode == "RGBA":
        background = Image.new("RGBA", pil_image.size, "white")
        pil_image = Image.alpha_composite(
            background, pil_image
        ).convert("RGB")
    else:
        pil_image = pil_image.convert("RGB")

    # Convert to grayscale
    pil_image = pil_image.convert("L")

    # Invert black writing on white background
    pil_image = ImageOps.invert(pil_image)

    # Resize to MNIST input size
    pil_image = ImageOps.fit(pil_image, (28, 28))

    # Normalize pixel values
    image_array = np.array(pil_image).astype("float32") / 255.0

    # Reshape for CNN model
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


# --------------------------------
# 3. PREDICTION FUNCTION
# --------------------------------

def predict_digit(drawing, upload):

    # Use drawing if available; otherwise use uploaded image
    image = drawing if drawing is not None else upload

    if image is None:
        return "Please draw or upload a digit first.", {}

    try:
        processed_image = prepare_image(image)

        if processed_image is None:
            return "Unable to process the image. Try again.", {}

        # Make prediction
        predictions = model.predict(
            processed_image,
            verbose=0
        )[0]

        digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        # Prepare top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]

        results = {
            str(int(i)): float(predictions[i])
            for i in top_indices
        }

        message = (
            f"Predicted Digit: {digit}\n"
            f"Confidence: {confidence:.2f}%"
        )

        return message, results

    except Exception as e:
        print("Prediction error:", str(e))
        return "Something went wrong while predicting. Please try again.", {}


# --------------------------------
# 4. CLEAR FUNCTION
# --------------------------------

def clear_all():
    return None, None, "", None


# --------------------------------
# 5. CUSTOM WEBSITE DESIGN
# --------------------------------

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #101c3a, #34205f, #5b267d) !important;
    color: white !important;
}

#main-title {
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-top: 10px;
}

#subtitle {
    text-align: center;
    color: #e0d7ff;
    font-size: 17px;
    margin-bottom: 25px;
}

.panel {
    background: white !important;
    border-radius: 18px !important;
    padding: 18px !important;
    color: #222 !important;
}

button {
    border-radius: 12px !important;
    font-weight: bold !important;
}

#predict-btn {
    background: linear-gradient(90deg, #6a5acd, #9b59b6) !important;
    color: white !important;
}

#clear-btn {
    background: #e8e8f0 !important;
    color: #333 !important;
}

#footer {
    text-align: center;
    color: #e0d7ff;
    margin-top: 25px;
    font-size: 14px;
}
"""


# --------------------------------
# 6. WEBSITE INTERFACE
# --------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple"
    ),
    css=custom_css
) as demo:

    gr.HTML("""
        <h1 id="main-title">
            ✨ Handwritten Digit Recognition System
        </h1>

        <p id="subtitle">
            Draw or upload a handwritten digit from 0 to 9
            and let AI recognize it!
        </p>
    """)

    with gr.Row():

        # Drawing section
        with gr.Column(elem_classes="panel"):

            gr.Markdown("### ✏️ Draw Your Digit")

            drawing = gr.Sketchpad(
                type="numpy",
                image_mode="RGB",
                canvas_size=(280, 280),
                brush=gr.Brush(
                    default_size=10,
                    colors=[
                        "#000000",
                        "#0000FF",
                        "#FF0000",
                        "#008000",
                        "#800080"
                    ],
                    default_color="#000000"
                ),
                label="Drawing Canvas"
            )

        # Upload section
        with gr.Column(elem_classes="panel"):

            gr.Markdown("### 📁 Upload an Image")

            upload = gr.Image(
                type="numpy",
                image_mode="RGB",
                label="Upload Handwritten Digit"
            )

    # Buttons
    with gr.Row():

        predict_btn = gr.Button(
            "🔍 Predict Digit",
            variant="primary",
            elem_id="predict-btn"
        )

        clear_btn = gr.Button(
            "🧹 Clear",
            elem_id="clear-btn"
        )

    # Prediction output
    with gr.Column(elem_classes="panel"):

        gr.Markdown("### 🎯 Recognition Result")

        message = gr.Textbox(
            label="Prediction Details",
            interactive=False,
            lines=2
        )

        output = gr.Label(
            num_top_classes=3,
            label="Top Predictions"
        )

    # Button actions
    predict_btn.click(
        fn=predict_digit,
        inputs=[drawing, upload],
        outputs=[message, output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[drawing, upload, message, output]
    )

    gr.HTML("""
        <div id="footer">
            <p>
                Built using Python, TensorFlow, CNN and Gradio
            </p>
            <p>
                AI-Based Handwritten Digit Recognition Project
            </p>
        </div>
    """)


# --------------------------------
# 7. LAUNCH APPLICATION
# --------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
