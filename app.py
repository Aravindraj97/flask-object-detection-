import os
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# -----------------------------
# Environment safety (Render)
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -----------------------------
# Load YOLO model ONCE
# -----------------------------
model = YOLO("yolov8n.pt")  # nano model only (safe for Render)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No selected file")

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save uploaded image
        file.save(image_path)

        # Run YOLO detection
        model.predict(
            source=image_path,
            save=True,
            project=RESULT_FOLDER,
            name="predict",
            exist_ok=True
        )

        # Output image path (must be inside static/)
        output_image = url_for(
            "static",
            filename=f"results/predict/{filename}"
        )

    return render_template("index.html", output_image=output_image)


# -----------------------------
# Local run (ignored by Render)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
