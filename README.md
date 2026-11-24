# RetinaSaarthi

**Explainable AI prototype for diabetic retinopathy screening support.**

RetinaSaarthi is an academic research project exploring how deep learning and explainable AI can support diabetic retinopathy (DR) screening, particularly in settings where access to specialist eye care is limited.

The project combines retinal image quality assessment, five-class DR grading, explainability experiments, and a web-based interface.

> **Research prototype disclaimer:** RetinaSaarthi is not a clinically validated medical device and must not be used as a substitute for examination or diagnosis by a qualified eye-care professional.

## Project objectives

1. **Image quality assessment:** Identify potentially unsuitable fundus images using image-quality measurements and quality-gate experiments.
2. **Five-class DR classification:** Explore EfficientNet-B0-based classification across No DR, Mild, Moderate, Severe, and Proliferative DR.
3. **Explainability:** Generate Grad-CAM visualizations and investigate their correspondence with available retinal lesion annotations.
4. **Cross-dataset robustness:** Evaluate model performance across different retinal image datasets.
5. **Screening workflow:** Develop a web interface and backend for image submission, results, and screening-support workflows.

Objectives are at different stages of implementation and evaluation. Refer to `PROGRESS.md` and the research reports for the current evidence and limitations.

## Technology stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI, Uvicorn |
| Machine learning | PyTorch, Torchvision, EfficientNet-B0 |
| Image processing | OpenCV, Pillow, NumPy |
| Data analysis | Pandas, scikit-learn |
| Authentication and database | Supabase |
| Testing | Pytest, Playwright |

## Repository structure

```text
RetinaSaarthi/
├── backend/             # FastAPI backend
├── frontend/            # React + TypeScript application
├── ml/
│   ├── classification/  # DR classification experiments
│   ├── quality/         # Image quality assessment
│   ├── explainability/  # Explainability experiments
│   └── evaluation/      # Model evaluation
├── datasets/            # Local datasets (not tracked by Git)
├── outputs/             # Model outputs and generated artifacts
├── reports/             # Research reports and evaluation evidence
├── scripts/             # Utility and verification scripts
├── supabase/            # Database migrations and functions
├── tests/               # Backend and project tests
├── docs/                # Project documentation
├── requirements.txt
├── PROGRESS.md
└── README.md
```

## Prerequisites

- Python 3.13
- Node.js and npm
- Git
- Supabase project credentials for features that require Supabase
- A CUDA-compatible NVIDIA GPU is recommended for model training, but is not required for basic application development.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vikas-g-10/RetinaSaarthi.git
cd RetinaSaarthi
```

### 2. Set up the Python environment

Create and activate a Python 3.13 virtual environment.

```bash
python3.13 -m venv .venv-313
source .venv-313/bin/activate
python -m pip install --upgrade pip
```

Install the backend and project dependencies:

```bash
python -m pip install -r requirements.txt
```

**Note:** PyTorch installation may need a platform-specific CPU or CUDA build. Select the appropriate build for your hardware if the pinned installation is unavailable.

### 3. Configure backend environment variables

```bash
cp .env.backend.example .env.backend
```

Edit `.env.backend` and provide your own Supabase project URL and publishable/anonymous key.

Do not commit real credentials or share private keys.

### 4. Start the backend

From the repository root, with the Python environment activated:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend API:

- Base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

### 5. Set up the frontend

Open a second terminal:

```bash
cd frontend
npm install
```

Create the frontend environment file:

```bash
cp .env.example .env
```

Configure the Supabase URL and publishable/anonymous key. Keep the API base URL pointed at the backend:

```text
VITE_API_BASE_URL=http://localhost:8000
```

Start the frontend development server:

```bash
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`.

## Testing and verification

### Backend tests

From the repository root, with the Python environment activated:

```bash
python -m pytest
```

### Frontend lint

```bash
cd frontend
npm run lint
```

### Frontend production build

```bash
npm run build
```

## Datasets and model artifacts

Datasets, trained checkpoints, and other large generated artifacts are intentionally excluded from Git.

Obtain datasets through their respective official sources and follow their licenses and terms of use. Place local dataset files in the appropriate directories under `datasets/`.

Do not assume that cloning this repository downloads the datasets or trained model checkpoints.

## Research and evaluation

Research scripts, dataset audits, preprocessing experiments, model evaluations, and supporting reports are organized under `ml/` and `reports/`.

The project distinguishes between:
- Training and validation performance.
- Evaluation on held-out data.
- Cross-dataset generalization.
- Exploratory explainability visualizations.
- Clinical validation, which has not been established.

Review the individual reports before interpreting any performance metric.

## Collaboration

1. Fork or clone the repository.
2. Create a separate feature branch.
3. Keep datasets, checkpoints, credentials, and generated temporary files out of commits.
4. Run relevant tests before submitting changes.
5. Submit a pull request describing the changes and verification performed.

## Current development status

See [`PROGRESS.md`](PROGRESS.md) for the development checklist, completed work, verification evidence, and known blockers.

## License

No project license has been specified yet. Do not assume that the repository or its contents are licensed for unrestricted reuse.
