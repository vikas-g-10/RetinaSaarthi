"""Verify full-size inference parity and Experiment 3 prediction agreement."""

from io import BytesIO
from pathlib import Path
import json

import pandas as pd
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from backend.main import app, require_researcher_dependency
from ml.classification.dataset import get_transforms, load_splits
from ml.classification.inference import InferenceService
from ml.classification.model import build_model

root = Path(__file__).resolve().parent
image_path = root / "datasets/aptos/train_images/000c1434d8d7.png"
checkpoint_path = root / "outputs/classification/experiment3/best_model.pth"
image_bytes = image_path.read_bytes()

service = InferenceService(checkpoint_path)
with Image.open(BytesIO(image_bytes)) as image:
    direct = service.predict(image)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model = build_model(pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with Image.open(BytesIO(image_bytes)) as image:
    tensor = get_transforms(training=False)(image.convert("RGB")).unsqueeze(0)
with torch.no_grad():
    independent = torch.softmax(model(tensor), dim=1)[0].tolist()

original_assess = main.assess_image
original_save = main.save_classification_result
main.assess_image = lambda _path: type(
    "Assessment", (), {
        "result": type("Result", (), {"decision": "PASS", "reason": "ok"})(),
        "metrics": None,
    }
)()
main.save_classification_result = lambda *_args, **_kwargs: "parity-result"

async def principal():
    return {"id": "parity-user", "role": "researcher", "token": "test-token"}

app.dependency_overrides[require_researcher_dependency] = principal
try:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/classification/predict",
            files={"file": (image_path.name, image_bytes, "image/png")},
            data={"patient_ref": "parity"},
        )
finally:
    app.dependency_overrides.clear()
    main.assess_image = original_assess
    main.save_classification_result = original_save

api_probs = response.json()["prediction"]["probabilities"]
print(json.dumps({
    "image": str(image_path.relative_to(root)),
    "bytes": len(image_bytes),
    "direct_service": direct.probabilities,
    "independent_original_transform": independent,
    "api_test_client_status": response.status_code,
    "api_test_client": api_probs,
    "max_abs_direct_vs_independent": max(abs(a-b) for a,b in zip(direct.probabilities, independent)),
    "max_abs_direct_vs_api": max(abs(a-b) for a,b in zip(direct.probabilities, api_probs)),
}, indent=2))

_, _, test = load_splits(seed=42)
predictions = pd.read_csv(root / "outputs/classification/experiment3/test_predictions.csv")
model_service = InferenceService(checkpoint_path)
rows = []
for _, row in predictions.head(20).iterrows():
    image_id = str(row["image_id"])
    path = root / "datasets/aptos/train_images" / f"{image_id}.png"
    with Image.open(path) as image:
        result = model_service.predict(image)
    split_row = test.dataframe.loc[test.dataframe["id_code"] == image_id]
    rows.append({
        "image_id": image_id,
        "mapped_file": str(path.relative_to(root)),
        "in_test_split_seed_42": not split_row.empty,
        "csv_predicted_grade": int(row["predicted_grade"]),
        "service_predicted_grade": result.predicted_class,
        "agree": int(row["predicted_grade"]) == result.predicted_class,
    })
print(json.dumps({"seed": 42, "rows": rows, "agreement": f"{sum(r['agree'] for r in rows)}/{len(rows)}"}, indent=2))
