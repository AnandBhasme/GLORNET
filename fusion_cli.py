"""
fusion_cli.py

Command-line interface for the diabetes and retinopathy fusion engine.

This provides a CLI alternative to the Streamlit app, useful when:
- Streamlit is not available or fails
- Running in headless environments
- Integrating into scripts or batch processing
- Automated workflows

Usage examples:
    # Basic usage with individual arguments
    python fusion_cli.py --fundus-image path/to/image.jpg --age 45 --bmi 25.5 --gender Male --glucose 120

    # Using JSON input file
    python fusion_cli.py --json-input patient_data.json

    # With only lab data (no image)
    python fusion_cli.py --age 50 --hba1c 6.5 --cholesterol 200 --hdl 50

    # Save output to file
    python fusion_cli.py --fundus-image image.jpg --age 45 --output results.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

# Import FusionEngine from fusion_app.py
from fusion_app import FusionEngine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GLORNET: Glucose and Ocular Retinal Neural Ensemble Technique - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with all data
  %(prog)s --fundus-image image.jpg --age 45 --bmi 25.5 --gender Male \\
           --glucose 120 --hba1c 6.8 --cholesterol 200

  # Using JSON input file
  %(prog)s --json-input patient_data.json

  # Lab data only (no retinal image)
  %(prog)s --age 50 --hba1c 6.5 --cholesterol 200 --hdl 50 --ldl 120

  # Save results to file
  %(prog)s --fundus-image image.jpg --age 45 --output results.txt
        """
    )

    # Input method (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--json-input",
        type=str,
        metavar="FILE",
        help="Path to JSON file containing all input data (see example format below)"
    )

    # Fundus image
    parser.add_argument(
        "--fundus-image",
        type=str,
        metavar="PATH",
        help="Path to fundus image file (JPG/PNG)"
    )

    # Shared fields (used by both lab and PIMA models)
    parser.add_argument("--age", type=str, help="Age (years)")
    parser.add_argument("--bmi", type=str, help="BMI (kg/m²)")

    # Lab features (CatBoost model)
    lab_group = parser.add_argument_group(
        "Laboratory-based model (CatBoost)",
        "These features are used by the CatBoost multiclass diabetes model"
    )
    lab_group.add_argument("--gender", type=str, choices=["Male", "Female"], help="Gender")
    lab_group.add_argument("--urea", type=str, help="Urea (mg/dL)")
    lab_group.add_argument("--creatinine", "--cr", type=str, dest="creatinine", help="Creatinine (mg/dL)")
    lab_group.add_argument("--hba1c", type=str, help="HbA1c (%)")
    lab_group.add_argument("--cholesterol", "--chol", type=str, dest="cholesterol", help="Cholesterol (mg/dL)")
    lab_group.add_argument("--triglycerides", "--tg", type=str, dest="triglycerides", help="Triglycerides (mg/dL)")
    lab_group.add_argument("--hdl", type=str, help="HDL (mg/dL)")
    lab_group.add_argument("--ldl", type=str, help="LDL (mg/dL)")
    lab_group.add_argument("--vldl", type=str, help="VLDL (mg/dL)")

    # PIMA features (LightGBM model)
    pima_group = parser.add_argument_group(
        "PIMA diabetes model (LightGBM)",
        "These features are used by the PIMA LightGBM binary diabetes model"
    )
    pima_group.add_argument("--pregnancies", type=str, help="Number of pregnancies")
    pima_group.add_argument("--glucose", type=str, help="Glucose (mg/dL)")
    pima_group.add_argument("--blood-pressure", "--bp", type=str, dest="blood_pressure", help="Blood Pressure (mmHg)")
    pima_group.add_argument("--skin-thickness", type=str, dest="skin_thickness", help="Skin Thickness (mm)")
    pima_group.add_argument("--insulin", type=str, help="Insulin (IU/mL)")
    pima_group.add_argument("--diabetes-pedigree-function", "--dpf", type=str, dest="dpf", help="Diabetes Pedigree Function")

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        metavar="FILE",
        help="Save results to file (default: print to stdout)"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format instead of human-readable text"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages (only show results)"
    )

    return parser.parse_args()


def load_json_input(json_path: str) -> Dict[str, Any]:
    """Load input data from JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON input file not found: {json_path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data


def build_lab_input_from_args(args) -> Dict[str, Any]:
    """Build lab input dictionary from command-line arguments."""
    lab_input = {}
    
    # Map command-line args to lab feature names
    mapping = {
        "gender": "Gender",
        "age": "AGE",
        "urea": "Urea",
        "creatinine": "Cr",
        "hba1c": "HbA1c",
        "cholesterol": "Chol",
        "triglycerides": "TG",
        "hdl": "HDL",
        "ldl": "LDL",
        "vldl": "VLDL",
        "bmi": "BMI",
    }
    
    for arg_name, feature_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            lab_input[feature_name] = value
    
    return lab_input


def build_pima_input_from_args(args) -> Dict[str, Any]:
    """Build PIMA input dictionary from command-line arguments."""
    pima_input = {}
    
    # Map command-line args to PIMA feature names
    mapping = {
        "pregnancies": "Pregnancies",
        "glucose": "Glucose",
        "blood_pressure": "BloodPressure",
        "skin_thickness": "SkinThickness",
        "insulin": "Insulin",
        "bmi": "BMI",
        "dpf": "DiabetesPedigreeFunction",
        "age": "Age",
    }
    
    for arg_name, feature_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            pima_input[feature_name] = value
    
    return pima_input


def load_fundus_image(image_path: str) -> Optional[Image.Image]:
    """Load fundus image from file."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Fundus image not found: {image_path}")
    
    try:
        img = Image.open(path)
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}") from e


def format_results_human_readable(result: Dict[str, Any]) -> str:
    """Format results as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("GLORNET: Glucose and Ocular Retinal Neural Ensemble Technique")
    lines.append("=" * 70)
    lines.append("")
    
    # Diabetes risk
    diab = result["diabetes_risk"]
    lines.append("METABOLIC (DIABETES) RISK")
    lines.append("-" * 70)
    
    pN = diab.get("p_non_diab", np.nan)
    pP = diab.get("p_prediab", np.nan)
    pY = diab.get("p_diab", np.nan)
    
    if not np.isnan(pN):
        lines.append(f"  Non-diabetic (N):     {pN*100:6.2f}%")
        lines.append(f"  Prediabetes (P):      {pP*100:6.2f}%")
        lines.append(f"  Diabetes (Y):         {pY*100:6.2f}%")
    else:
        lines.append("  Insufficient systemic data to estimate diabetes risk.")
    
    lines.append("")
    
    # Retinal risk
    retina = result.get("retinal_risk")
    lines.append("RETINAL (EYE) STATUS")
    lines.append("-" * 70)
    
    if retina is not None:
        p_anyDR = retina.get("p_anyDR", np.nan)
        p_refDR = retina.get("p_refDR", np.nan)
        p_ciDME = retina.get("p_ciDME", np.nan)
        
        lines.append(f"  Any DR:                     {p_anyDR*100:6.2f}%")
        lines.append(f"  Referable DR (≥ moderate):  {p_refDR*100:6.2f}%")
        lines.append(f"  Center-involved DME:        {p_ciDME*100:6.2f}%")
        
        # Detailed DR probabilities
        dr_probs = retina.get("dr_probs", [])
        if len(dr_probs) == 5:
            lines.append("")
            lines.append("  DR Grade Probabilities:")
            lines.append(f"    Grade 0 (No DR):        {dr_probs[0]*100:6.2f}%")
            lines.append(f"    Grade 1 (Mild):         {dr_probs[1]*100:6.2f}%")
            lines.append(f"    Grade 2 (Moderate):     {dr_probs[2]*100:6.2f}%")
            lines.append(f"    Grade 3 (Severe):       {dr_probs[3]*100:6.2f}%")
            lines.append(f"    Grade 4 (Proliferative): {dr_probs[4]*100:6.2f}%")
        
        # Detailed DME probabilities
        dme_probs = retina.get("dme_probs", [])
        if len(dme_probs) == 3:
            lines.append("")
            lines.append("  DME Grade Probabilities:")
            lines.append(f"    Grade 0 (No DME):       {dme_probs[0]*100:6.2f}%")
            lines.append(f"    Grade 1 (Non-CI DME):   {dme_probs[1]*100:6.2f}%")
            lines.append(f"    Grade 2 (CI DME):       {dme_probs[2]*100:6.2f}%")
    else:
        lines.append("  No fundus image provided - retinal risk cannot be estimated.")
    
    lines.append("")
    
    # Clinical interpretation
    lines.append("CLINICAL INTERPRETATION")
    lines.append("-" * 70)
    interpretation = result.get("interpretation", "")
    # Remove markdown formatting for plain text
    interpretation_clean = interpretation.replace("**", "").replace("\n", "\n  ")
    lines.append(f"  {interpretation_clean}")
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def format_results_json(result: Dict[str, Any]) -> str:
    """Format results as JSON."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    
    result_serializable = convert_to_serializable(result)
    return json.dumps(result_serializable, indent=2)


def print_progress(msg: str, quiet: bool = False):
    """Print progress message if not in quiet mode."""
    if not quiet:
        print(msg, file=sys.stderr)


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Load input data
    lab_input = {}
    pima_input = {}
    fundus_image = None
    
    if args.json_input:
        # Load from JSON file
        print_progress(f"Loading input data from {args.json_input}...", args.quiet)
        json_data = load_json_input(args.json_input)
        
        # Extract lab, pima, and image path from JSON
        lab_input = json_data.get("lab", {})
        pima_input = json_data.get("pima", {})
        image_path = json_data.get("fundus_image")
        
        if image_path:
            print_progress(f"Loading fundus image from {image_path}...", args.quiet)
            fundus_image = load_fundus_image(image_path)
    else:
        # Build from command-line arguments
        lab_input = build_lab_input_from_args(args)
        pima_input = build_pima_input_from_args(args)
        
        if args.fundus_image:
            print_progress(f"Loading fundus image from {args.fundus_image}...", args.quiet)
            fundus_image = load_fundus_image(args.fundus_image)
    
    # Initialize fusion engine
    print_progress("Loading fusion engine models...", args.quiet)
    try:
        engine = FusionEngine()
    except Exception as e:
        print(f"ERROR: Failed to load fusion engine: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Run analysis
    print_progress("Running analysis...", args.quiet)
    try:
        result = engine.analyze(
            lab_input=lab_input,
            pima_input=pima_input,
            fundus_image=fundus_image,
        )
    except Exception as e:
        print(f"ERROR: Analysis failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Format and output results
    if args.json_output:
        output_text = format_results_json(result)
    else:
        output_text = format_results_human_readable(result)
    
    if args.output:
        # Save to file
        with open(args.output, 'w') as f:
            f.write(output_text)
        print_progress(f"Results saved to {args.output}", args.quiet)
    else:
        # Print to stdout
        print(output_text)


if __name__ == "__main__":
    main()

