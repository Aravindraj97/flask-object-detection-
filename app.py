import os
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# -------------------------
# Environment safety (Render)
# -------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------
# App setup
# -------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------------------------
# Load YOLO model once
# -------------------------
model = YOLO("yolov8n.pt")   # lightweight model

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "No file uploaded"
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "No selected file"
            return render_template("index.html", error=error)

        # Save uploaded image
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        # Run YOLO prediction
        results = model.predict(
            source=image_path,
            save=True,
            project=RESULT_FOLDER,
            name="predict",
            exist_ok=True
        )

        # YOLO saves output with same filename under /predict/
        predicted_filename = os.path.basename(image_path)
        predicted_path = os.path.join(RESULT_FOLDER, "predict", predicted_filename)

        # Convert path for HTML <img>
        output_image = url_for("static", filename=f"results/predict/{predicted_filename}")

        return render_template("index.html", output_image=output_image, error=error)

    return render_template("index.html", output_image=output_image, error=error)


# -------------------------
# Local run
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
