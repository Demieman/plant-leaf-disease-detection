
from pathlib import Path
from io import BytesIO

import cv2
import joblib
import numpy as np
import streamlit as st

from PIL import Image, UnidentifiedImageError

from features import extract_features
from segment import analyze


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

EXPECTED_CLASSES = 10
EXPECTED_FEATURE_VERSION = 2
EXPECTED_FEATURE_COUNT = 116
MAX_FILE_SIZE = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

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


# --------------------------------------------------
# 2. LOAD AND VALIDATE THE MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    possible_paths = [
        BASE_DIR / "models" / "leaf_svm_10.joblib",
        BASE_DIR / "leaf_svm_10.joblib"
    ]

    model_path = next(
        (path for path in possible_paths if path.exists()),
        None
    )

    if model_path is None:
        raise FileNotFoundError(
            "Model file not found. Expected "
            "'leaf_svm_10.joblib' in the repository root "
            "or inside the models folder."
        )

    model_bundle = joblib.load(model_path)

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
            f"Unsupported feature version: {feature_version}"
        )

    if feature_count != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but found {feature_count}."
        )

    if svm_model.n_features_in_ != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "The model has an unexpected feature count."
        )

    return svm_model, class_names, model_path


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🌿 Plant Leaf Disease Detection System")

st.write(
    "Upload a tomato leaf image to classify its disease "
    "using a trained Support Vector Machine (SVM) model. "
    "The application also provides K-Means color segmentation "
    "and a visual analysis of potentially discolored regions."
)

st.divider()


# --------------------------------------------------
# 4. LOAD MODEL
# --------------------------------------------------

try:
    with st.spinner("Loading the trained SVM model..."):
        svm_model, class_names, model_path = load_model()

    st.success("SVM model loaded successfully.")

except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


# --------------------------------------------------
# 5. SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("Analysis Settings")

clusters = st.sidebar.slider(
    "Number of K-Means clusters",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

st.sidebar.info(
    "K-Means controls the number of color groups "
    "used in the segmentation visualization. "
    "It does not change the SVM disease classes."
)

st.sidebar.markdown("---")
st.sidebar.write("**Classifier:** Support Vector Machine")
st.sidebar.write("**Classes:** 10")
st.sidebar.write("**Feature count:** 116")
st.sidebar.write("**Feature version:** 2")


# --------------------------------------------------
# 6. IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP. Maximum size: 8 MB."
)


# --------------------------------------------------
# 7. PROCESS UPLOADED IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    try:

        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > MAX_FILE_SIZE:
            raise ValueError(
                "The uploaded file exceeds the 8 MB limit."
            )

        # Verify the uploaded image
        with Image.open(BytesIO(image_bytes)) as check_image:
            check_image.verify()

        # Reopen and convert to RGB
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            image = uploaded_image.convert("RGB")

        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "Image resolution is too large. "
                "Please upload a smaller image."
            )

        rgb = np.asarray(image, dtype=np.uint8)

        # OpenCV functions expect BGR
        image_bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

        st.subheader("Uploaded Image")
        st.image(
            rgb,
            caption="Original tomato leaf image",
            use_container_width=True
        )

        # ------------------------------------------
        # 8. SVM FEATURE EXTRACTION
        # ------------------------------------------

        with st.spinner("Extracting image features..."):

            extracted = extract_features(image_bgr)

            if extracted.size != EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"Feature mismatch: expected "
                    f"{EXPECTED_FEATURE_COUNT}, "
                    f"but extracted {extracted.size}."
                )

            feature_vector = np.asarray(
                extracted,
                dtype=np.float32
            ).reshape(1, -1)

        # ------------------------------------------
        # 9. SVM CLASSIFICATION
        # ------------------------------------------

        with st.spinner("Classifying the leaf image..."):

            prediction = svm_model.predict(
                feature_vector
            )[0]

            predicted_class = str(prediction)

            display_name = CLASS_LABELS.get(
                predicted_class,
                predicted_class
            )

        # ------------------------------------------
        # 10. CLASS PROBABILITY
        # ------------------------------------------

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

                class_index = int(matching_indices[0])

                confidence = float(
                    probabilities[class_index] * 100
                )

        # ------------------------------------------
        # 11. DISPLAY CLASSIFICATION RESULT
        # ------------------------------------------

        st.divider()
        st.header("SVM Classification Result")

        st.success(f"Predicted class: **{display_name}**")

        if confidence is not None:
            st.metric(
                "Model probability for predicted class",
                f"{confidence:.2f}%"
            )

        st.caption(
            "The probability is the model's output, "
            "not a guarantee that the prediction is correct."
        )

        # Show the three highest class probabilities
        if probabilities is not None:

            st.subheader("Class Probability Details")

            sorted_indices = np.argsort(
                probabilities
            )[::-1][:3]

            for index in sorted_indices:

                raw_name = str(svm_model.classes_[index])

                label = CLASS_LABELS.get(
                    raw_name,
                    raw_name
                )

                probability = float(
                    probabilities[index] * 100
                )

                st.write(f"**{label}: {probability:.2f}%**")
                st.progress(
                    min(max(probability / 100, 0.0), 1.0)
                )

        # ------------------------------------------
        # 12. K-MEANS VISUAL ANALYSIS
        # ------------------------------------------

        st.divider()
        st.header("K-Means Color Segmentation")

        with st.spinner(
            f"Analyzing the leaf using {clusters} clusters..."
        ):

            result = analyze(
                image_bgr,
                clusters=clusters
            )

        # ------------------------------------------
        # 13. DISPLAY PROCESSED IMAGES
        # ------------------------------------------

        st.subheader("Image Analysis Results")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                result["original"],
                caption="Original Image",
                use_container_width=True
            )

            st.image(
                result["segmented"],
                caption="K-Means Segmentation",
                use_container_width=True
            )

        with col2:

            st.image(
                result["overlay"],
                caption="Potential Discoloration Overlay",
                use_container_width=True
            )

            st.image(
                result["abnormal_mask"],
                caption="Potential Discoloration Mask",
                use_container_width=True
            )

        # ------------------------------------------
        # 14. DISPLAY PIXEL STATISTICS
        # ------------------------------------------

        st.divider()
        st.subheader("Visual Analysis Statistics")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Leaf pixels",
            f'{result["leaf_pixels"]:,}'
        )

        metric2.metric(
            "Flagged pixels",
            f'{result["affected_pixels"]:,}'
        )

        metric3.metric(
            "Potential discoloration",
            f'{result["affected_percent"]:.2f}%'
        )

        st.info(
            "The discoloration percentage is a color-based "
            "heuristic estimate of potentially yellow or brown "
            "pixels. It is not a validated disease severity "
            "measurement and does not independently diagnose disease."
        )

        st.success("Image analysis completed.")

    except (
        ValueError,
        UnidentifiedImageError,
        OSError,
        cv2.error
    ) as exc:

        st.error(f"Image analysis error: {exc}")

    except Exception as exc:

        st.error(
            "An unexpected error occurred. "
            f"Details: {exc}"
        )

else:

    st.info(
        "Please upload a tomato leaf image to begin the analysis."
    )


# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Plant Leaf Disease Detection System | "
    "SVM Classification + K-Means Color Segmentation"
)
