# GLORNET

**Glucose and Ocular Retinal Neural Ensemble Technique**

A multi-modal AI system for diabetes risk assessment and diabetic retinopathy/macular edema detection, combining laboratory data, clinical features, and retinal fundus imaging.

## Overview

GLORNET integrates three complementary machine learning models:

1. **CatBoost Multiclass Model** - Lab-based diabetes classification (Non-diabetic / Prediabetes / Diabetes)
2. **LightGBM PIMA Model** - Binary diabetes risk assessment using PIMA Indian Diabetes dataset features
3. **Deep Learning Retina Model** - Multi-task CNN for diabetic retinopathy (DR) grading (0-4) and diabetic macular edema (DME) risk (0-2)

The fusion engine combines all three models to provide comprehensive metabolic and ocular health assessment.

## Features

- Multi-modal health assessment combining systemic and ocular data
- Probability calibration for reliable risk estimates
- Support for missing/incomplete data
- Two interfaces: Command-line (CLI) and web-based (Streamlit)
- Handles various input formats (individual arguments or JSON files)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for retinal model inference)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd GLORNET

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

#### Basic Usage

```bash
# Full analysis with fundus image and lab data
python fusion_cli.py --fundus-image path/to/fundus.jpg \
                     --age 45 --bmi 25.5 --gender Male \
                     --glucose 120 --hba1c 6.8

# Using JSON input file
python fusion_cli.py --json-input patient_data.json

# Lab data only (no retinal image)
python fusion_cli.py --age 50 --hba1c 6.5 --cholesterol 200 --hdl 50

# Save results to file
python fusion_cli.py --fundus-image image.jpg --age 45 --output results.txt
```

#### JSON Input Example

Create a file `patient_data.json`:

```json
{
  "fundus_image": "path/to/fundus_image.jpg",
  "lab": {
    "Gender": "Male",
    "AGE": 45,
    "Urea": 35,
    "Cr": 1.0,
    "HbA1c": 6.5,
    "Chol": 200,
    "TG": 150,
    "HDL": 45,
    "LDL": 120,
    "VLDL": 30,
    "BMI": 27.5
  },
  "pima": {
    "Pregnancies": 0,
    "Glucose": 120,
    "BloodPressure": 80,
    "SkinThickness": 20,
    "Insulin": 100,
    "BMI": 27.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 45
  }
}
```

### Streamlit Web Interface

```bash
streamlit run fusion_app.py --server.port 5998
```

Open your browser to the URL shown (typically `http://localhost:5998`).

## Model Architecture

### Retinal Model
- **Backbone**: MobileNetV3-Large (timm)
- **Input size**: 320x320 RGB
- **Preprocessing**: Fundus crop, enhancement, normalization
- **Outputs**: 
  - DR grades 0-4 (None, Mild, Moderate, Severe, Proliferative)
  - DME grades 0-2 (None, Non-center-involved, Center-involved)
- **Calibration**: Temperature scaling for improved probability estimates

### Laboratory Model (CatBoost)
- **Type**: Gradient boosted trees (multiclass)
- **Features**: Gender, Age, Urea, Creatinine, HbA1c, Cholesterol, Triglycerides, HDL, LDL, VLDL, BMI
- **Output**: Non-diabetic (N) / Prediabetes (P) / Diabetes (Y)
- **Calibration**: Isotonic regression

### PIMA Model (LightGBM)
- **Type**: Gradient boosted trees (binary)
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Output**: Diabetes risk probability
- **Calibration**: Isotonic regression

## Datasets

The models were trained on the following datasets:

### IDRiD (Indian Diabetic Retinopathy Image Dataset)
<!-- Add dataset link here -->
- **Link**: [IEEEDataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
- **Usage**: Retinal DR/DME grading
- **Size**: 516 training images, 103 test images

### Diabetes Laboratory Dataset
<!-- Add dataset link here -->
- **Link**: [Mendeley](https://data.mendeley.com/datasets/wj9rwkp9c2/1)
- **Usage**: Lab-based diabetes classification

### PIMA Indians Diabetes Database
<!-- Add dataset link here -->
- **Link**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Usage**: Binary diabetes risk prediction

## Clinical Interpretation

The system provides:
- **Metabolic Status**: Probabilities for non-diabetic, prediabetes, and diabetes
- **Retinal Status**: Risk of any DR, referable DR (moderate or worse), and center-involved DME
- **Systemic-Retinal Consistency Check**: Flags potential mismatches between systemic and retinal findings

**Important**: This is a research tool and should not replace professional medical diagnosis. All results should be interpreted by qualified healthcare professionals.
