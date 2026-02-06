import os

# limit CPU threads for Render free tier
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# load model ONCE (global)
model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # force CPU inference
            results = model(image_path, device="cpu")

            result_image_path = os.path.join(RESULT_FOLDER, file.filename)
            results[0].save(filename=result_image_path)

            #  freeing disk space
            os.remove(image_path)

            return render_template(
                "index.html",
                result_image=result_image_path
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
