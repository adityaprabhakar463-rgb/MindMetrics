# AI Study Predictor

## Overview

The **AI Study Predictor** is a full-stack web application designed to help students optimize their study habits. It uses machine learning models to predict productivity scores and assess burnout risk based on various factors such as study hours, sleep quality, mood, and exam proximity.

## Tech Stack

- **Backend**: Python, Flask, Scikit-learn, Pandas, NumPy
- **Frontend**: React, Vite, TypeScript, Tailwind CSS, Shadcn UI
- **Machine Learning**: Linear Regression (Productivity), Logistic Regression (Burnout)

## Project Structure

```
AI_STUDY_PREDICTOR/
├── ai-productivity-predictor/  # React Frontend
├── data/                       # Dataset storage
├── models/                     # Trained ML models
├── src/                        # Data processing & training scripts
│   ├── data_processing.py      # Generates synthetic data
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── train_model.py          # Trains Productivity Model
│   └── train_burnout_model.py  # Trains Burnout Model
├── app.py                      # Flask API Backend
├── requirements.txt            # Backend dependencies
└── run_project.sh              # Helper script to run full stack
```

## Setup & Installation

### Prerequisites

- Node.js & npm (for Frontend)
- Python 3.8+ (for Backend)

### 1. Backend Setup

1.  Navigate to the project root:
    ```bash
    cd /path/to/AI_STUDY_PREDICTOR
    ```
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd ai-productivity-predictor
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```

## Usage

### Training Models

Before running the API, you need to generate data and train the machine learning models.

1.  **Generate Data**:
    ```bash
    python src/data_processing.py
    ```
    This creates `data/study_data.csv`.

2.  **Train Models**:
    ```bash
    python src/train_model.py          # Trains Productivity Model
    python src/train_burnout_model.py  # Trains Burnout Model
    ```
    Models will be saved in the `models/` directory.

### Running the Application

You can start both the backend and frontend using the provided helper script:

```bash
./run_project.sh
```

Alternatively, run them separately:

**Backend (Port 5001):**
```bash
python app.py
```

**Frontend (Port 5173 or similar):**
```bash
cd ai-productivity-predictor
npm run dev
```

## API Endpoints

The Flask API runs at `http://localhost:5001`.

### 1. Health Check
- **Endpoint**: `GET /`
- **Description**: Returns API status and available endpoints.

### 2. Predict Productivity
- **Endpoint**: `POST /api/predict/productivity`
- **Body**:
  ```json
  {
      "Study Hours": 6.5,
      "Sleep Hours": 7.0,
      "Mood": 4,
      "Distraction": 2,
      "Difficulty": 3,
      "Exam Proximity": 15
  }
  ```
- **Response**: Returns a productivity score (0-100) and interpretation.

### 3. Predict Burnout
- **Endpoint**: `POST /api/predict/burnout`
- **Body**: Same as above.
- **Response**: Returns burnout risk (0 or 1), probability, and interpretation.
