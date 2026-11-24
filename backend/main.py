
"""FastAPI application for the RetinaSaarthi research prototype."""

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import json
import os
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ml.classification.inference import ModelUnavailable, get_inference_service
from ml.quality.service import assess_image


# ------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env.backend",
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]


# ------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="RetinaSaarthi API",
    description=(
        "Backend API for the RetinaSaarthi diabetic "
        "retinopathy screening research prototype."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MAX_UPLOAD_SIZE = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8000
MAX_IMAGE_PIXELS = 25_000_000

ALLOWED_TYPES = {
    "image/jpeg": (".jpg", "JPEG"),
    "image/png": (".png", "PNG"),
}

ALLOWED_DECISIONS = {
    "PASS",
    "RECAPTURE",
    "MANUAL_REVIEW",
}

bearer = HTTPBearer(auto_error=False)


# ------------------------------------------------------------
# JSON serialization helper
# ------------------------------------------------------------

def json_default(value):
    """Convert supported NumPy scalar values to JSON-compatible values."""

    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


# ------------------------------------------------------------
# Diagnostic helper
# ------------------------------------------------------------

def log_http_error(stage: str, exc: HTTPError) -> None:
    """Log Supabase HTTP error details without logging credentials."""

    try:
        response_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
    except Exception:
        response_body = "Unable to read error response."

    # Limit diagnostic output.
    response_body = response_body[:500]

    print(
        f"[RetinaSaarthi] {stage} failed: "
        f"HTTP {exc.code}, "
        f"reason={exc.reason}, "
        f"response={response_body}"
    )


# ------------------------------------------------------------
# Authentication and role verification
# ------------------------------------------------------------

def require_researcher(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    """
    Validate the Supabase session and active profile.

    Only active researcher and administrator accounts can
    submit quality assessments.
    """

    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    if not supabase_url or not supabase_anon_key:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is not configured.",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Sign in to use this endpoint.",
        )

    token = credentials.credentials

    headers = {
        "apikey": supabase_anon_key,
        "Authorization": f"Bearer {token}",
    }

    stage = "Supabase Auth user verification"

    try:
        # Step 1: Verify the access token with Supabase Auth.

        auth_request = Request(
            f"{supabase_url}/auth/v1/user",
            headers=headers,
        )

        with urlopen(auth_request, timeout=8) as response:
            user = json.loads(response.read())

        if not user.get("email_confirmed_at"):
            raise HTTPException(
                status_code=403,
                detail="Verify your email before using this endpoint.",
            )

        if not user.get("id"):
            raise ValueError("Supabase Auth response did not contain a user ID.")

        # Step 2: Retrieve the user's active profile and role.

        stage = "Supabase profile lookup"

        query = urlencode({
            "id": f"eq.{user['id']}",
            "select": "role,is_active",
        })

        profile_request = Request(
            f"{supabase_url}/rest/v1/profiles?{query}",
            headers=headers,
        )

        with urlopen(profile_request, timeout=8) as response:
            profiles = json.loads(response.read())

    except HTTPException:
        raise

    except HTTPError as exc:
        log_http_error(stage, exc)

        if stage == "Supabase Auth user verification":
            if exc.code in (400, 401, 403):
                raise HTTPException(
                    status_code=401,
                    detail="Your session is invalid or expired.",
                ) from exc

        if stage == "Supabase profile lookup":
            if exc.code in (401, 403):
                raise HTTPException(
                    status_code=403,
                    detail="Unable to access your account profile.",
                ) from exc

        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable.",
        ) from exc

    except URLError as exc:
        print(
            f"[RetinaSaarthi] {stage} connection failed: "
            f"{type(exc.reason).__name__}: {exc.reason}"
        )

        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable.",
        ) from exc

    except (
        TimeoutError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(
            f"[RetinaSaarthi] {stage} failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable.",
        ) from exc

    # Step 3: Verify that the profile exists and is active.

    if not profiles:
        raise HTTPException(
            status_code=403,
            detail="No active profile was found for this account.",
        )

    profile = profiles[0]

    if not profile.get("is_active"):
        raise HTTPException(
            status_code=403,
            detail="This account is inactive.",
        )

    role = profile.get("role")

    if role not in ("admin", "researcher"):
        raise HTTPException(
            status_code=403,
            detail="Your role cannot submit assessments.",
        )

    return {
        "id": user["id"],
        "role": role,
        "token": token,
    }


async def require_researcher_dependency(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    """Async dependency adapter for Python 3.14/AnyIO test and ASGI runtimes."""
    return require_researcher(credentials)


# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "project": "RetinaSaarthi",
        "status": "running",
        "version": "0.2.0",
    }


# ------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "retinasaarthi-backend",
    }


# ------------------------------------------------------------
# Assessment history persistence
# ------------------------------------------------------------

def save_assessment(
    principal: dict,
    assessment_result: dict,
    metrics: dict | None,
) -> str | None:
    """
    Save assessment metadata to Supabase.

    The uploaded retinal image is never sent to Supabase.
    Only the result, safe filename, reason and metrics are saved.
    """

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Assessment completed but history storage "
                "is not configured."
            ),
        )

    record = {
        "user_id": principal["id"],
        "filename": "fundus-image",
        "decision": assessment_result["decision"],
        "reason": assessment_result["reason"],
        "metrics": metrics,
    }

    try:
        payload = json.dumps(
            record,
            default=json_default,
            allow_nan=False,
        ).encode("utf-8")

    except (TypeError, ValueError) as exc:
        print(
            "[RetinaSaarthi] Assessment serialization failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Assessment history could not be serialized.",
        ) from exc

    persist_request = Request(
        f"{SUPABASE_URL}/rest/v1/assessments",
        data=payload,
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {principal['token']}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        method="POST",
    )

    try:
        with urlopen(persist_request, timeout=8) as response:
            response_data = json.loads(response.read() or b"[]")
            if response_data and isinstance(response_data, list):
                return response_data[0].get("id")
            return None

    except HTTPError as exc:
        log_http_error("Supabase assessment persistence", exc)

        raise HTTPException(
            status_code=503,
            detail=(
                "Assessment completed but could not be saved "
                "to your history."
            ),
        ) from exc


def save_classification_result(
    principal: dict,
    assessment_result: dict,
    metrics: dict | None,
    classification_result: dict,
) -> str | None:
    """Atomically persist assessment and classification metadata with the caller JWT."""
    payload = json.dumps(
        {
            "p_assessment": {
                "user_id": principal["id"],
                "filename": "fundus-image",
                "decision": assessment_result["decision"],
                "reason": assessment_result["reason"],
                "metrics": metrics,
            },
            "p_classification": classification_result,
        },
        default=json_default,
        allow_nan=False,
    ).encode("utf-8")
    request = Request(
        f"{SUPABASE_URL}/rest/v1/rpc/create_classification_result",
        data=payload,
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {principal['token']}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=8) as response:
            response_data = json.loads(response.read() or b"null")
            return response_data if isinstance(response_data, str) else None
    except HTTPError as exc:
        log_http_error("Supabase classification result persistence", exc)
        if exc.code in (400, 404):
            raise HTTPException(
                status_code=502,
                detail="Classification results are unavailable because the database migration has not been applied.",
            ) from exc
        raise HTTPException(status_code=502, detail="Classification result could not be saved atomically.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(status_code=502, detail="Classification result could not be saved atomically.") from exc


# ------------------------------------------------------------
# Fundus image quality assessment endpoint
# ------------------------------------------------------------

@app.post("/api/v1/quality/assess")
async def assess_fundus_image(
    file: UploadFile = File(...),
    _principal: dict = Depends(require_researcher_dependency),
):
    """
    Upload a fundus image and return its quality assessment.

    Processing:
    1. Validate declared file type.
    2. Enforce upload size limit.
    3. Validate image dimensions and actual format.
    4. Decode the image.
    5. Run the existing quality assessment service.
    6. Save assessment metadata to Supabase.
    7. Return the assessment response.

    Original uploaded images are never permanently stored.
    """

    contents, filename = await _read_upload(file)
    image, suffix = _validate_and_decode_upload(contents, filename, file.content_type)

    # Step 5: Run quality assessment.

    # Use a temporary file. The original image is deleted
    # when this context exits.

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / f"fundus_image{suffix}"
        image_path.write_bytes(contents)

        assessment = assess_image(str(image_path))

    # Step 6: Prepare assessment result.

    decision = getattr(
        assessment.result.decision,
        "value",
        assessment.result.decision,
    )

    if decision not in ALLOWED_DECISIONS:
        raise HTTPException(
            status_code=500,
            detail=(
                "The quality assessment returned "
                "an unsupported decision."
            ),
        )

    assessment_data = {
        "decision": decision,
        "reason": str(assessment.result.reason),
    }

    metrics_data = (
        asdict(assessment.metrics)
        if assessment.metrics is not None
        else None
    )

    # Step 7: Build response.

    result = {
        "project": "RetinaSaarthi",
        "assessment": assessment_data,
        "metrics": metrics_data,
        "clinical_status": (
            "Research prototype only. Not clinically validated "
            "and not intended for autonomous diagnosis."
        ),
    }

    # Step 8: Persist metadata before returning.

    save_assessment(
        principal=_principal,
        assessment_result=assessment_data,
        metrics=metrics_data,
    )

    # Step 9: Return response.

    return result


class PredictionResponse(BaseModel):
    predicted_class: int = Field(ge=0, le=4)
    predicted_label: str
    probabilities: list[float] = Field(min_length=5, max_length=5)
    model_name: str
    model_version: str


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    contents = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="The uploaded image exceeds the 20 MB limit.")
    return bytes(contents), file.filename or ""


def _validate_and_decode_upload(
    contents: bytes,
    filename: str,
    declared_content_type: str | None = None,
) -> tuple[np.ndarray, str]:
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 20 MB limit.")
    if declared_content_type and declared_content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported.")
    filename_suffix = Path(filename).suffix.lower()
    magic_type = None
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        magic_type = (".png", "PNG")
    elif contents.startswith(b"\xff\xd8\xff"):
        magic_type = (".jpg", "JPEG")
    elif filename_suffix not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported.")
    else:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    suffix, expected_format = magic_type
    if declared_content_type and ALLOWED_TYPES[declared_content_type][1] != expected_format:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    try:
        with Image.open(BytesIO(contents)) as pil_image:
            width, height = pil_image.size
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the allowed limit.")
            if pil_image.format != expected_format:
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
            pil_image.verify()
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=422, detail="The uploaded file could not be decoded.")
        return image, suffix
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc


@app.get("/api/v1/model/status")
async def model_status():
    return get_inference_service().status()


@app.post("/api/v1/classification/predict", response_model=dict)
async def predict_classification(
    file: UploadFile = File(...),
    patient_ref: str | None = Form(default=None, max_length=120),
    principal: dict = Depends(require_researcher_dependency),
):
    if patient_ref is not None:
        if len(patient_ref) > 120 or any(ord(char) < 32 for char in patient_ref):
            raise HTTPException(status_code=400, detail="Patient reference must be a non-empty pseudonymous identifier.")
        patient_ref = patient_ref.strip()
        if not patient_ref:
            raise HTTPException(status_code=400, detail="Patient reference must be a non-empty pseudonymous identifier.")

    contents, filename = await _read_upload(file)
    image, suffix = _validate_and_decode_upload(contents, filename, file.content_type)
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / f"fundus_image{suffix}"
        image_path.write_bytes(contents)
        assessment = assess_image(str(image_path))

    decision = getattr(assessment.result.decision, "value", assessment.result.decision)
    metrics_data = asdict(assessment.metrics) if assessment.metrics is not None else None
    quality_block = {
        "decision": decision,
        "reason": str(assessment.result.reason),
        "metrics": metrics_data,
    }
    # Classification is allowed only after an explicit PASS. MANUAL_REVIEW
    # must not silently fall through to model inference.
    if decision != "PASS":
        return {"project": "RetinaSaarthi", "quality": quality_block, "prediction": None}

    try:
        with Image.open(BytesIO(contents)) as pil_image:
            prediction = get_inference_service().predict(pil_image)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    classification_result_id = save_classification_result(
        principal,
        {"decision": decision, "reason": str(assessment.result.reason)},
        metrics_data,
        {
        "user_id": principal["id"],
        "predicted_class": prediction.predicted_class,
        "predicted_label": prediction.predicted_label,
        "probabilities": prediction.probabilities,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
        "patient_ref": patient_ref,
        },
    )
    if not classification_result_id:
        raise HTTPException(
            status_code=503,
            detail="Classification completed but the saved result did not return an identifier.",
        )
    return {
        "project": "RetinaSaarthi",
        "result_id": classification_result_id,
        "quality": quality_block,
        "prediction": PredictionResponse(
            predicted_class=prediction.predicted_class,
            predicted_label=prediction.predicted_label,
            probabilities=prediction.probabilities,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
        ).model_dump(),
    }
