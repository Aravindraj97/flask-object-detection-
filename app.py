import os
from pathlib import Path
from time import time

from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# -------------------------
# Environment safety (Render/limited CPU)
# -------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------
# App setup
# -------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Optional: limit allowed extensions (helps with clearer errors)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# -------------------------
# Load YOLO model once
# -------------------------
model = YOLO("yolov8n.pt")  # CPU-friendly nano model

# -------------------------
# Helpers
# -------------------------
def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def find_predicted_image(save_dir: Path, stem: str) -> Path | None:
    """
    YOLO writes annotated images into save_dir using the input file's stem,
    but the extension may differ from the uploaded one (often .jpg).
    We glob for any "{stem}.*" and pick the first match.
    """
    candidates = list(save_dir.glob(f"{stem}.*"))
    if candidates:
        return candidates[0]

    # Fallbacks if glob didn't catch anything (ultra-defensive)
    for ext in (".jpg", ".jpeg", ".png"):
        p = save_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None
    error = None
    cache_buster = int(time())

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", error=error)

        file = request.files["image"]
        if file.filename == "":
            error = "No selected file."
            return render_template("index.html", error=error)

        if not is_allowed(file.filename):
            error = f"Unsupported file type: {Path(file.filename).suffix.lower()}"
            return render_template("index.html", error=error)

        # Save uploaded image
        filename = secure_filename(file.filename)
        image_path = UPLOAD_FOLDER / filename
        file.save(image_path)

        try:
            # Run YOLO prediction (saves annotated output in RESULT_FOLDER/name)
            results = model.predict(
                source=str(image_path),
                save=True,
                project=str(RESULT_FOLDER),
                name="predict",
                exist_ok=True
            )

            # Where YOLO actually saved annotated files
            # Typically: static/results/predict
            save_dir = Path(results[0].save_dir)
            stem = image_path.stem

            predicted_path = find_predicted_image(save_dir, stem)
            if not predicted_path or not predicted_path.exists():
                error = "Prediction completed, but the output file was not found."
                return render_template("index.html", error=error)

            # Build URL under /static
            # save_dir is ".../static/results/predict"
            # We need the portion relative to "/static"
            # => results/predict/<predicted_filename>
            rel_under_static = Path("results") / save_dir.name / predicted_path.name

            # Cache-busting query param prevents stale browser cache
            output_image = url_for("static", filename=str(rel_under_static).replace("\\", "/")) + f"?v={cache_buster}"

            return render_template("index.html", output_image=output_image, error=error)

        except Exception as e:
            error = f"Prediction failed: {e}"
            return render_template("index.html", error=error)

    return render_template("index.html", output_image=output_image, error=error)


# -------------------------
# Local run
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
