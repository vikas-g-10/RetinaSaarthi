
from io import BytesIO
from pathlib import Path
import struct
import zlib
import json
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app, require_researcher, require_researcher_dependency


client = TestClient(app)


@pytest.fixture(autouse=True)
def authenticated_api_requests(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return b""

    monkeypatch.setenv("SUPABASE_URL", "https://project.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-test-key")
    monkeypatch.setattr("backend.main.urlopen", lambda *_args, **_kwargs: Response())
    async def fake_principal():
        return {"id": "test-user", "role": "researcher", "token": "test-token"}

    app.dependency_overrides[require_researcher_dependency] = fake_principal
    yield
    app.dependency_overrides.clear()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "datasets"
    / "aptos"
    / "train_images"
    / "000c1434d8d7.png"
)


def make_png_with_dimensions(width, height):
    """Create a small PNG with specified dimensions in its IHDR metadata."""

    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")

    contents = bytearray(buffer.getvalue())

    # PNG IHDR image dimensions start at byte offset 16.
    struct.pack_into(">II", contents, 16, width, height)

    # Update the IHDR checksum after modifying its dimensions.
    checksum = zlib.crc32(contents[12:29]) & 0xFFFFFFFF
    struct.pack_into(">I", contents, 29, checksum)

    return bytes(contents)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "RetinaSaarthi"
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_quality_api_rejects_missing_bearer_token(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-test-key")
    with pytest.raises(HTTPException) as error:
        require_researcher(None)
    assert error.value.status_code == 401


def test_quality_api_fails_closed_without_supabase_configuration(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    with pytest.raises(HTTPException) as error:
        require_researcher(None)
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    ("role", "is_active", "expected_status"),
    [("reviewer", True, 403), ("researcher", False, 403), ("admin", True, 200)],
)
def test_quality_api_enforces_profile_role_and_active_status(monkeypatch, role, is_active, expected_status):
    monkeypatch.setenv("SUPABASE_URL", "https://project.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-test-key")

    class Response:
        def __init__(self, data): self.data = json.dumps(data).encode()
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return self.data

    @contextmanager
    def fake_urlopen(request, timeout):
        assert timeout == 8
        if request.full_url.endswith("/auth/v1/user"):
            yield Response({"id": "user-1", "email_confirmed_at": "2026-01-01T00:00:00Z"})
        else:
            yield Response([{"role": role, "is_active": is_active}])

    monkeypatch.setattr("backend.main.urlopen", fake_urlopen)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
    if expected_status == 200:
        assert require_researcher(credentials)["role"] == "admin"
    else:
        with pytest.raises(HTTPException) as error:
            require_researcher(credentials)
        assert error.value.status_code == expected_status


def test_valid_fundus_image_upload():
    with SAMPLE_IMAGE.open("rb") as image:
        response = client.post(
            "/api/v1/quality/assess",
            files={
                "file": (
                    "fundus.png",
                    image,
                    "image/png",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["assessment"]["decision"] == "MANUAL_REVIEW"
    assert data["metrics"] is not None
    assert data["metrics"]["width"] == 3216
    assert data["metrics"]["height"] == 2136
    assert 0 < data["metrics"]["field_coverage_frame_percent"] <= 100
    assert "not clinically validated" in data["clinical_status"].lower()


def test_unsupported_file_type():
    response = client.post(
        "/api/v1/quality/assess",
        files={
            "file": (
                "document.txt",
                b"sample text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_empty_upload():
    response = client.post(
        "/api/v1/quality/assess",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_invalid_image_content():
    response = client.post(
        "/api/v1/quality/assess",
        files={
            "file": (
                "invalid.png",
                b"this is not a real image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_oversized_upload():
    oversized_content = b"x" * (20 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/quality/assess",
        files={
            "file": (
                "large.png",
                oversized_content,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_image_exceeding_maximum_dimension():
    contents = make_png_with_dimensions(8001, 1)

    response = client.post(
        "/api/v1/quality/assess",
        files={
            "file": (
                "wide.png",
                contents,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_image_exceeding_maximum_pixel_count():
    contents = make_png_with_dimensions(6000, 5000)

    response = client.post(
        "/api/v1/quality/assess",
        files={
            "file": (
                "large_dimensions.png",
                contents,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_image_format_must_match_declared_content_type():
    with SAMPLE_IMAGE.open("rb") as image:
        response = client.post(
            "/api/v1/quality/assess",
            files={
                "file": (
                    "incorrect_type.jpg",
                    image,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 400
