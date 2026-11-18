
from pathlib import Path
from io import BytesIO
import base64

import cv2
import joblib
import numpy as np

from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from segment import analyze
from features import extract_features


# ----------------------------------------
# 1. INITIALIZE FLASK APPLICATION
# ----------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "leaf_svm_10.joblib"

EXPECTED_CLASSES = 10
EXPECTED_FEATURE_VERSION = 2
EXPECTED_FEATURE_COUNT = 116


# ----------------------------------------
# 2. LOAD AND VALIDATE SVM MODEL
# ----------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run train_svm.py first."
    )

model_bundle = joblib.load(MODEL_PATH)

svm_model = model_bundle["model"]
class_names = model_bundle["class_names"]

feature_version = model_bundle.get("feature_version")
feature_count = model_bundle.get(
    "feature_count",
    getattr(svm_model, "n_features_in_", None)
)

if len(class_names) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"but found {len(class_names)}."
    )

if feature_version != EXPECTED_FEATURE_VERSION:
    raise ValueError(
        f"Unsupported feature version: {feature_version}."
    )

if feature_count != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"but found {feature_count}."
    )

if svm_model.n_features_in_ != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        "The loaded SVM model has an unexpected feature count."
    )

print("\n========== SVM MODEL LOADED ==========")
print("Application directory:", BASE_DIR)
print("Model path:", MODEL_PATH)
print("Feature version:", feature_version)
print("Feature count:", feature_count)
print("Model classes:", svm_model.classes_)
print("======================================\n")


# ----------------------------------------
# 3. FRIENDLY CLASS NAMES
# ----------------------------------------

CLASS_LABELS = {
    "Tomato__Target_Spot": "Tomato Target Spot",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_healthy": "Healthy Tomato Leaf",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite":
        "Tomato Spider Mites"
}


# ----------------------------------------
# 4. CONVERT IMAGE TO BASE64
# ----------------------------------------

def to_data_url(array, mode="RGB"):

    output = BytesIO()

    Image.fromarray(array, mode=mode).save(
        output,
        format="PNG"
    )

    encoded = base64.b64encode(
        output.getvalue()
    ).decode("ascii")

    return "data:image/png;base64," + encoded


# ----------------------------------------
# 5. MAIN APPLICATION ROUTE
# ----------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        upload = request.files.get("leaf_image")

        try:

            # Validate uploaded file
            if not upload or not upload.filename:
                raise ValueError(
                    "Please select a leaf image."
                )

            if not upload.filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                raise ValueError(
                    "Upload a JPG, JPEG, PNG, or WEBP image."
                )

            # Verify uploaded image
            img = Image.open(upload.stream)
            img.verify()

            upload.stream.seek(0)

            img = Image.open(
                upload.stream
            ).convert("RGB")

            if img.width * img.height > 20_000_000:
                raise ValueError(
                    "Image resolution is too large."
                )

            # --------------------------------
            # 6. CONVERT RGB TO BGR
            # --------------------------------

            rgb = np.asarray(
                img,
                dtype=np.uint8
            )

            image_bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )

            # --------------------------------
            # 7. SVM FEATURE EXTRACTION
            # --------------------------------

            extracted = extract_features(
                image_bgr
            )

            if extracted.size != feature_count:
                raise ValueError(
                    f"Feature mismatch: expected "
                    f"{feature_count}, but extracted "
                    f"{extracted.size}."
                )

            feature_vector = np.asarray(
                extracted,
                dtype=np.float32
            ).reshape(1, -1)

            # --------------------------------
            # 8. SVM CLASSIFICATION
            # --------------------------------

            prediction = svm_model.predict(
                feature_vector
            )[0]

            predicted_class = str(prediction)

            display_name = CLASS_LABELS.get(
                predicted_class,
                predicted_class
            )

            # --------------------------------
            # 9. CLASS PROBABILITY
            # --------------------------------

            confidence = None
            probabilities = None

            if hasattr(svm_model, "predict_proba"):

                probabilities = svm_model.predict_proba(
                    feature_vector
                )[0]

                matching_indices = np.where(
                    svm_model.classes_ == prediction
                )[0]

                if len(matching_indices) > 0:

                    class_index = int(
                        matching_indices[0]
                    )

                    confidence = round(
                        float(
                            probabilities[class_index] * 100
                        ),
                        2
                    )

            # --------------------------------
            # 10. FLASK SVM DIAGNOSTICS
            # --------------------------------

            print("\n========== FLASK SVM DIAGNOSTICS ==========")

            print("Application directory:", BASE_DIR)
            print("Model path:", MODEL_PATH)
            print("Uploaded filename:", upload.filename)

            print(
                "Uploaded image dimensions:",
                image_bgr.shape
            )

            print(
                "Extracted feature count:",
                extracted.size
            )

            print(
                "Feature vector shape:",
                feature_vector.shape
            )

            print(
                "Feature vector data type:",
                feature_vector.dtype
            )

            print(
                "Flask raw prediction:",
                predicted_class
            )

            print(
                "Displayed prediction:",
                display_name
            )

            print(
                "Predicted class probability:",
                confidence
            )

            if probabilities is not None:

                print("\nClass probabilities:")

                for index in np.argsort(
                    probabilities
                )[::-1]:

                    print(
                        f"{svm_model.classes_[index]}: "
                        f"{probabilities[index] * 100:.2f}%"
                    )

            print("===========================================\n")


            # 11. K-MEANS VISUAL ANALYSIS
        

            try:

                clusters = int(
                    request.form.get("clusters", 3)
                )

            except (TypeError, ValueError):

                raise ValueError(
                    "Invalid number of K-Means clusters."
                )

            if not 2 <= clusters <= 10:
                raise ValueError(
                    "K-Means clusters must be between 2 and 10."
                )

            analyzed = analyze(
                image_bgr,
                clusters
            )

            # --------------------------------
            # 12. COMBINE RESULTS
            # --------------------------------

            result = {

                "prediction": predicted_class,

                "display_name": display_name,

                "confidence": confidence,

                "affected_percent":
                    analyzed["affected_percent"],

                "leaf_pixels":
                    analyzed["leaf_pixels"],

                "affected_pixels":
                    analyzed["affected_pixels"],

                "clusters":
                    analyzed["clusters"]
            }

            # --------------------------------
            # 13. PROCESSED IMAGES
            # --------------------------------

            for key in (
                "original",
                "segmented",
                "overlay"
            ):

                result[key] = to_data_url(
                    analyzed[key]
                )

            result["abnormal_mask"] = to_data_url(
                analyzed["abnormal_mask"],
                "L"
            )

        except (
            ValueError,
            UnidentifiedImageError,
            OSError,
            cv2.error
        ) as exc:

            error = str(exc)

            print("\nAPPLICATION ERROR:", error)

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ----------------------------------------
# 14. FILE SIZE ERROR HANDLER
# ----------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def too_large(error):

    return render_template(
        "index.html",
        result=None,
        error="File exceeds the 8 MB upload limit."
    ), 413


# ----------------------------------------
# 15. RUN APPLICATION
# ----------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
