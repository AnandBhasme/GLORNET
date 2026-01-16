"""
fusion_app.py

Fusion engine for:
1) CatBoost multiclass (N / P / Y) based on lab data
2) LightGBM PIMA diabetes model (binary)
3) Retinal multi-task CNN (DR grade 0–4, DME grade 0–2)

+ Simple Streamlit web frontend for interactive use.

Medically motivated hierarchy:
- Lab-based CatBoost is primary for diabetes status.
- PIMA LightGBM is auxiliary / risk-only (especially when labs missing).
- Retina model independently grades DR / DME.
- Fusion layer combines all 3 into a coherent report.
"""

import io
from pathlib import Path
from typing import Dict, Any, Optional

import cv2


import numpy as np
import pandas as pd
from PIL import Image

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder

# Import your retinal model + constants + image helpers
# Make sure idrid_fundus.py is in the same folder.
try:
    from Scripts.idrid_fundus import (
        MultiTaskEffNet,
        MODEL_NAME,
        NUM_DR_CLASSES,
        NUM_DME_CLASSES,
        IMG_SIZE,
        retina_enhance,
        fundus_bbox_square,
    )
except ImportError as e:
    print(f"ERROR: Failed to import idrid_fundus module: {e}")
    print("Please ensure idrid_fundus.py is in the same directory as fusion_app.py")
    raise
except Exception as e:
    print(f"ERROR: Unexpected error importing idrid_fundus: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================
# Temperature scaler for retinal logits
# ============================================================

class TemperatureScaler(nn.Module):
    """
    Simple temperature scaling module:
    logits_scaled = logits / T, where T = exp(log_temp) > 0
    """
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))  # T = 1.0 initially

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T


# ============================================================
# Fusion Engine
# ============================================================

class FusionEngine:
    """
    Fusion engine that:
      - Loads calibrated CatBoost, PIMA LightGBM, and retinal models.
      - Handles missing fields (NaN) for tree models.
      - Produces probabilities and a clinical-style interpretation.
    """

    # Paths – adjust if your files are named differently
    CATBOOST_PATH = "models/catboost_diabetes_multiclass_calibrated.pkl"
    CATBOOST_LABEL_PATH = "models/catboost_diabetes_label_encoder.pkl"
    PIMA_PATH = "models/pima_lgbm_diabetes_calibrated.pkl"
    RETINA_CAL_CKPT = "models/best_idrid_simple_calibrated.pt"

    # Feature definitions
    LAB_FEATURES = [
    "Gender", "AGE", "Urea", "Cr", "HbA1c",
    "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"
    ]


    PIMA_FEATURES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    # In PIMA training, zero was treated as "missing" for these:
    PIMA_ZERO_AS_MISSING = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]

    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # ---- Load CatBoost (lab) ----
            if not Path(self.CATBOOST_PATH).exists():
                raise FileNotFoundError(f"CatBoost model file not found: {self.CATBOOST_PATH}")
            if not Path(self.CATBOOST_LABEL_PATH).exists():
                raise FileNotFoundError(f"CatBoost label encoder file not found: {self.CATBOOST_LABEL_PATH}")
            
            self.catboost = joblib.load(self.CATBOOST_PATH)
            self.catboost_label_encoder: LabelEncoder = joblib.load(self.CATBOOST_LABEL_PATH)

            # ---- Load PIMA LightGBM ----
            if not Path(self.PIMA_PATH).exists():
                raise FileNotFoundError(f"PIMA model file not found: {self.PIMA_PATH}")
            
            self.pima_model = joblib.load(self.PIMA_PATH)

            # ---- Load retina model + temperatures ----
            if not Path(self.RETINA_CAL_CKPT).exists():
                raise FileNotFoundError(f"Retina checkpoint file not found: {self.RETINA_CAL_CKPT}")
            
            self.retina_model = MultiTaskEffNet(
                backbone_name=MODEL_NAME,
                num_dr=NUM_DR_CLASSES,
                num_dme=NUM_DME_CLASSES,
                drop_rate=0.4
            ).to(self.device)

            ckpt = torch.load(self.RETINA_CAL_CKPT, map_location=self.device)
            self.retina_model.load_state_dict(ckpt["model_state_dict"])

            self.temp_dr = TemperatureScaler().to(self.device)
            self.temp_dme = TemperatureScaler().to(self.device)
            self.temp_dr.load_state_dict(ckpt["temp_dr_state_dict"])
            self.temp_dme.load_state_dict(ckpt["temp_dme_state_dict"])

            self.retina_model.eval()
            self.temp_dr.eval()
            self.temp_dme.eval()
        except Exception as e:
            error_msg = f"Error loading models: {str(e)}\n\n"
            error_msg += f"Please ensure the following files exist:\n"
            error_msg += f"  - {self.CATBOOST_PATH}\n"
            error_msg += f"  - {self.CATBOOST_LABEL_PATH}\n"
            error_msg += f"  - {self.PIMA_PATH}\n"
            error_msg += f"  - {self.RETINA_CAL_CKPT}\n"
            raise RuntimeError(error_msg) from e

    # --------------------------------------------------------
    # Tabular preprocessing
    # --------------------------------------------------------

    def _to_float_or_nan(self, v: Any) -> float:
        """Convert UI value (possibly empty string) to float or NaN."""
        if v is None:
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan

    def build_lab_df(self, lab_input: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Build a single-row DataFrame for CatBoost lab model.
        lab_input keys: Gender, Age, Urea, Creatinine, HbA1c, Cholesterol, Triglycerides, HDL, LDL, VLDL, BMI
        Missing numeric values become NaN.
        """
        if not lab_input:
            return None

        row = {}
        for col in self.LAB_FEATURES:
            if col == "Gender":
                # Allow "Male"/"Female" or empty
                g = lab_input.get("Gender", "").strip()
                if g == "":
                    row["Gender"] = np.nan  # CatBoost can handle missing
                else:
                    row["Gender"] = g
            else:
                row[col] = self._to_float_or_nan(lab_input.get(col))

        df = pd.DataFrame([row], columns=self.LAB_FEATURES)
        return df

    def build_pima_df(self, pima_input: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Build a single-row DataFrame for PIMA LightGBM model.
        Missing numeric values become NaN. For columns where 0 was treated as missing in training,
        we also convert zeros to NaN.
        """
        if not pima_input:
            return None

        row = {}
        for col in self.PIMA_FEATURES:
            row[col] = self._to_float_or_nan(pima_input.get(col))

        df = pd.DataFrame([row], columns=self.PIMA_FEATURES)

        # Convert 0 -> NaN for specific cols (as done in training)
        for col in self.PIMA_ZERO_AS_MISSING:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)

        return df

    # --------------------------------------------------------
    # Retina image preprocessing
    # --------------------------------------------------------

    def preprocess_fundus_image(self, img: Image.Image) -> torch.Tensor:
        """
        Convert a PIL fundus image to a normalized tensor of shape (1, 3, IMG_SIZE, IMG_SIZE),
        using the same basic preprocessing as training (crop, enhance, resize).
        """
        # PIL -> RGB numpy
        rgb = np.array(img.convert("RGB"))

        # Crop + enhance + resize (same style as training)
        rgb = fundus_bbox_square(rgb, pad_ratio=0.01)
        rgb = retina_enhance(rgb)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # HWC -> CHW, normalize to [0,1]
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (0, 1, 2)) if x.ndim == 3 else x  # safety
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, 3, H, W)
        return x_tensor.to(self.device)

    # --------------------------------------------------------
    # Model calls
    # --------------------------------------------------------

    def _diabetes_from_lab_pima(
        self,
        lab_df: Optional[pd.DataFrame],
        pima_df: Optional[pd.DataFrame],
        alpha: float = 0.7,
    ) -> Dict[str, float]:
        """
        Combine CatBoost (lab) + PIMA LightGBM into final N/P/Y probabilities.
        alpha is the weight for lab; (1-alpha) for PIMA risk.
        """
        pN = pP = pY = None
        p_diab_pima = None

        has_lab = lab_df is not None
        has_pima = pima_df is not None

        if has_lab:
            probs_lab = self.catboost.predict_proba(lab_df)[0]  # order: [encoded0, encoded1, encoded2]
            # Map to N/P/Y using label encoder
            # label_encoder.classes_ is [N, P, Y]
            # But just in case, we use inverse_transform
            enc_idxs = np.arange(len(self.catboost_label_encoder.classes_))
            lab_names = self.catboost_label_encoder.inverse_transform(enc_idxs)
            mapping = {name: probs_lab[i] for i, name in enumerate(lab_names)}

            pN = mapping.get("N", 0.0)
            pP = mapping.get("P", 0.0)
            pY = mapping.get("Y", 0.0)

        if has_pima:
            probs_pima = self.pima_model.predict_proba(pima_df)[0]
            # binary: [p_non_diab, p_diab]
            p_diab_pima = float(probs_pima[1])

        # Fusion logic
        if has_lab and has_pima:
            # Labs dominate, PIMA is auxiliary
            p_diab_final = alpha * (pY or 0.0) + (1 - alpha) * (p_diab_pima or 0.0)
            p_pred_final = pP or 0.0
        elif has_lab:
            p_diab_final = pY or 0.0
            p_pred_final = pP or 0.0
        elif has_pima:
            p_diab_final = p_diab_pima or 0.0
            p_pred_final = 0.0
        else:
            return {"p_non_diab": np.nan, "p_prediab": np.nan, "p_diab": np.nan}

        # Ensure probabilities sum <= 1
        p_non_final = max(0.0, 1.0 - p_diab_final - p_pred_final)
        # Clip to [0,1]
        p_non_final = float(np.clip(p_non_final, 0.0, 1.0))
        p_pred_final = float(np.clip(p_pred_final, 0.0, 1.0))
        p_diab_final = float(np.clip(p_diab_final, 0.0, 1.0))

        # Renormalize slightly if needed
        s = p_non_final + p_pred_final + p_diab_final
        if s > 0:
            p_non_final /= s
            p_pred_final /= s
            p_diab_final /= s

        return {
            "p_non_diab": p_non_final,
            "p_prediab": p_pred_final,
            "p_diab": p_diab_final,
        }

    def _retina_probs(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Run retina model on a single preprocessed tensor and return DR & DME probabilities.
        """
        with torch.no_grad():
            logits_dr, logits_dme = self.retina_model(img_tensor)

            logits_dr = self.temp_dr(logits_dr)
            logits_dme = self.temp_dme(logits_dme)

            probs_dr = F.softmax(logits_dr, dim=1).cpu().numpy()[0]   # shape (5,)
            probs_dme = F.softmax(logits_dme, dim=1).cpu().numpy()[0] # shape (3,)

        # Useful summaries
        p_anyDR = 1.0 - probs_dr[0]
        p_refDR = float(probs_dr[2] + probs_dr[3] + probs_dr[4])  # moderate or worse
        p_ciDME = float(probs_dme[2])  # central-involved DME risk (grade 2)

        return {
            "dr_probs": probs_dr,
            "dme_probs": probs_dme,
            "p_anyDR": float(p_anyDR),
            "p_refDR": p_refDR,
            "p_ciDME": p_ciDME,
        }

    # --------------------------------------------------------
    # High-level API
    # --------------------------------------------------------

    def analyze(
        self,
        lab_input: Dict[str, Any],
        pima_input: Dict[str, Any],
        fundus_image: Optional[Image.Image],
    ) -> Dict[str, Any]:
        """
        Main entry point.
        lab_input  : dict of lab features
        pima_input : dict of PIMA features
        fundus_image: PIL Image or None

        Returns a dict with:
          - diabetes_risk: dict of N/P/Y probabilities
          - retinal_risk: dict of DR/DME probabilities and derived scores (if image provided)
          - interpretation: string summary
        """
        lab_df = self.build_lab_df(lab_input)
        pima_df = self.build_pima_df(pima_input)

        diabetes_risk = self._diabetes_from_lab_pima(lab_df, pima_df)

        retinal_risk = None
        if fundus_image is not None:
            img_tensor = self.preprocess_fundus_image(fundus_image)
            retinal_risk = self._retina_probs(img_tensor)

        interpretation = self._build_interpretation(diabetes_risk, retinal_risk)

        return {
            "diabetes_risk": diabetes_risk,
            "retinal_risk": retinal_risk,
            "interpretation": interpretation,
        }

    def _build_interpretation(
        self,
        diab: Dict[str, float],
        retina: Optional[Dict[str, Any]],
    ) -> str:
        """
        Build a clinically sensible text summary.
        """
        lines = []

        pN = diab.get("p_non_diab", np.nan)
        pP = diab.get("p_prediab", np.nan)
        pY = diab.get("p_diab", np.nan)

        # Diabetes status
        if not np.isnan(pN):
            lines.append("**Metabolic (diabetes) status**")
            lines.append(
                f"- Probability non-diabetic (N): **{pN*100:.1f}%**\n"
                f"- Probability prediabetes (P): **{pP*100:.1f}%**\n"
                f"- Probability diabetes (Y): **{pY*100:.1f}%**"
            )

            if pY >= 0.6:
                lines.append(
                    "- Interpretation: High probability of diabetes based on available data. "
                    "Formal diagnosis should rely on standard criteria (fasting glucose, OGTT, or HbA1c) "
                    "and clinical judgement."
                )
            elif pP >= 0.3:
                lines.append(
                    "- Interpretation: Intermediate risk / possible prediabetes. "
                    "Lifestyle modification and monitoring of glycaemia are advisable."
                )
            else:
                lines.append(
                    "- Interpretation: Currently low probability of diabetes from the available systemic data."
                )
        else:
            lines.append(
                "**Metabolic (diabetes) status**\n"
                "- Insufficient systemic data to estimate diabetes risk."
            )

        # Retinal status
        if retina is not None:
            p_anyDR = retina["p_anyDR"]
            p_refDR = retina["p_refDR"]
            p_ciDME = retina["p_ciDME"]

            lines.append("\n**Retinal (eye) status**")
            lines.append(
                f"- Probability of any diabetic retinopathy: **{p_anyDR*100:.1f}%**\n"
                f"- Probability of referable DR (moderate or worse): **{p_refDR*100:.1f}%**\n"
                f"- Probability of center-involved DME: **{p_ciDME*100:.1f}%**"
            )

            if p_refDR >= 0.5 or p_ciDME >= 0.5:
                lines.append(
                    "- Interpretation: High risk of sight-threatening diabetic eye disease. "
                    "Prompt evaluation by a retina specialist is recommended."
                )
            elif p_anyDR >= 0.35:
                lines.append(
                    "- Interpretation: Evidence of early / mild diabetic retinopathy. "
                    "Optimisation of systemic risk factors and regular retinal screening are important."
                )
            else:
                lines.append(
                    "- Interpretation: Low probability of diabetic retinopathy or macular edema on this image. "
                    "Routine screening should still follow local guidelines for people with diabetes."
                )

            # Consistency check
            if pY < 0.3 and (p_refDR > 0.4 or p_ciDME > 0.4):
                lines.append(
                    "\n**Systemic–retinal mismatch:** Retinal findings suggest significant diabetic eye disease "
                    "despite low estimated systemic diabetes probability. This may represent:\n"
                    "- Undiagnosed or previously treated diabetes,\n"
                    "- Other causes of retinopathy (e.g., hypertensive, vascular occlusions), or\n"
                    "- Limitations of the systemic risk models.\n"
                    "Clinical correlation and full metabolic workup are advised."
                )
        else:
            lines.append(
                "\n**Retinal (eye) status**\n"
                "- No fundus image was provided, so retinopathy/macular edema risk cannot be estimated by the model."
            )



        return "\n".join(lines)


# ============================================================
# Simple Streamlit frontend
# ============================================================

def run_streamlit_app():
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Install with `pip install streamlit` and run:\n"
              "    streamlit run fusion_app.py")
        return

    # Test imports first
    try:
        import sys
        print("Testing imports...", file=sys.stderr)
        print("Testing imports...")
    except Exception as e:
        st.error(f"Error during initialization: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
        return

    st.set_page_config(page_title="Diabetes & Retinopathy Fusion AI", layout="wide")

    st.title("GLORNET: Glucose and Ocular Retinal Neural Ensemble Technique")

    st.markdown(
        "Upload a retinal fundus image and enter systemic data. "
        "The system will combine **lab-based CatBoost**, **PIMA LightGBM**, and a **retinal CNN** "
        "to estimate diabetes status and diabetic eye disease risk."
    )

    @st.cache_resource
    def load_engine_cached():
        # Don't use st.error/st.stop inside cached function - raise exception instead
        try:
            engine = FusionEngine()
            return engine
        except Exception as e:
            # Store error message to be displayed outside cached function
            raise e

    # Try to load engine and handle errors outside cached function
    try:
        with st.spinner("Loading models... This may take a moment."):
            engine = load_engine_cached()
    except FileNotFoundError as e:
        st.error(f"**Model file not found:**\n\n{str(e)}\n\nPlease ensure all model files are in the correct location.")
        st.stop()
        return
    except Exception as e:
        import traceback
        st.error(f"**Failed to load fusion engine:**\n\n{str(e)}\n\n```\n{traceback.format_exc()}\n```")
        st.stop()
        return

    # ---- Shared fields for Age and BMI ----
    st.subheader("Shared Patient Information")
    col_shared1, col_shared2 = st.columns(2)
    with col_shared1:
        age = st.text_input("Age (years)", "", key="shared_age")
    with col_shared2:
        bmi = st.text_input("BMI (kg/m²)", "", key="shared_bmi")

    st.markdown("---")

    col_img, col_sys1, col_sys2 = st.columns([1.2, 1, 1])

    # ---- Image column ----
    with col_img:
        st.subheader("Fundus image")
        img_file = st.file_uploader("Upload a color fundus image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        fundus_image = None
        if img_file is not None:
            fundus_image = Image.open(io.BytesIO(img_file.read()))
            st.image(fundus_image, caption="Uploaded fundus image", use_container_width=True)

    # ---- Systemic data: labs ----
    with col_sys1:
        st.subheader("Laboratory-based diabetes model (CatBoost)")
        gender = st.selectbox("Gender", options=["", "Male", "Female"], index=0)
        urea = st.text_input("Urea (mg/dL)", "")
        creat = st.text_input("Creatinine (mg/dL)", "")
        hba1c = st.text_input("HbA1c (%)", "")
        chol = st.text_input("Cholesterol (mg/dL)", "")
        tg = st.text_input("Triglycerides (mg/dL)", "")
        hdl = st.text_input("HDL (mg/dL)", "")
        ldl = st.text_input("LDL (mg/dL)", "")
        vldl = st.text_input("VLDL (mg/dL)", "")

    # ---- Systemic data: PIMA ----
    with col_sys2:
        st.subheader("PIMA diabetes model (LightGBM)")
        preg = st.text_input("Pregnancies", "")
        glucose = st.text_input("Glucose (mg/dL)", "")
        bp = st.text_input("Blood Pressure (mmHg)", "")
        skin = st.text_input("Skin Thickness (mm)", "")
        insulin = st.text_input("Insulin (IU/mL)", "")
        dpf = st.text_input("Diabetes Pedigree Function", "")

    st.markdown("---")
    if st.button("Run fused analysis"):
        with st.spinner("Running models..."):
            lab_input = {
                "Gender": gender,
                "AGE": age,
                "Urea": urea,
                "Cr": creat,
                "HbA1c": hba1c,
                "Chol": chol,
                "TG": tg,
                "HDL": hdl,
                "LDL": ldl,
                "VLDL": vldl,
                "BMI": bmi,
            }


            pima_input = {
                "Pregnancies": preg,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }

            result = engine.analyze(
                lab_input=lab_input,
                pima_input=pima_input,
                fundus_image=fundus_image,
            )

        st.subheader("Results")
        diab = result["diabetes_risk"]
        retina = result["retinal_risk"]

        if not np.isnan(diab.get("p_non_diab", np.nan)):
            st.write("**Metabolic risk (N / P / Y):**")
            st.write(
                f"- Non-diabetic (N): **{diab['p_non_diab']*100:.1f}%**  \n"
                f"- Prediabetes (P): **{diab['p_prediab']*100:.1f}%**  \n"
                f"- Diabetes (Y): **{diab['p_diab']*100:.1f}%**"
            )
        else:
            st.write("**Metabolic risk:** insufficient data.")

        if retina is not None:
            st.write("**Retinal risk:**")
            st.write(
                f"- Any DR: **{retina['p_anyDR']*100:.1f}%**  \n"
                f"- Referable DR (≥ moderate): **{retina['p_refDR']*100:.1f}%**  \n"
                f"- Centre-involved DME: **{retina['p_ciDME']*100:.1f}%**"
            )
        else:
            st.write("**Retinal risk:** no fundus image provided.")

        st.markdown("---")
        st.subheader("Clinical-style interpretation")
        st.markdown(result["interpretation"])


if __name__ == "__main__":
    # Run as a Streamlit app:
    #   streamlit run fusion_app.py
    try:
        run_streamlit_app()
    except Exception as e:
        import traceback
        print("=" * 60)
        print("FATAL ERROR during app startup:")
        print("=" * 60)
        print(f"{type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        raise
