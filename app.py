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
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "no file uploaded"
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error="No selected file"
            return render_template("index.html", error= error)

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save uploaded image
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        # Run YOLO detection
        results = model.predict(
            source=image_path,
            save=True,
            project=RESULT_FOLDER,
            name="predict",
            exist_ok=True
        )
        saved_path = results[0].path

        relative_path = os.path.relpath(saved_path,os.path.join(BASE_DIR,"static"))
        output_image = url_for(
            "static",
            filename=relative_path
        )

    return render_template("index.html", output_image=output_image,error=error)


# -----------------------------
# Local run (ignored by Render)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
