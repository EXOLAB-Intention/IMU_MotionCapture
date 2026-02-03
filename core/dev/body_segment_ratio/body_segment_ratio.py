import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict
from pathlib import Path
import sys


class BodySegmentRatioPredictor:
    """
    A class for predicting body segment ratios.
    
    Uses direct measurement data to predict trunk, thigh, shank, and foot ratios
    from height and foot length using linear regression.
    """
    
    # Class constants
    FILE_PATH = ".\\8차 인체치수조사(2020~24)_치수데이터(공개용).xlsx"
    SHEET_NAME = "(1~2차년도) 직접측정"
    
    # Column mapping
    HEIGHT_COL = "002. 키"
    SHOULDER_COL = "005. 어깨높이"
    WAIST_COL = "009. 허리높이 "
    HIP_COL = "012. 엉덩이높이 "
    KNEE_COL = "014. 무릎높이 "
    ANKLE_COL = "015. 가쪽복사높이 "
    FOOT_COL = "122. 발직선길이 "
    
    def __init__(self, file_path: str = None):
        """
        Load data and train linear regression models.
        
        Parameters
        ----------
        file_path : str, optional
            Path to the Excel file. If None, use default path.
        """
        if file_path:
            self.FILE_PATH = file_path
            
        self.models: Dict[str, LinearRegression] = {}
        self.df = None
        
        # Load data and train models
        self._load_data()
        self._train_models()
    
    def _load_data(self):
        """Load and preprocess data."""
        # ========================================
        # 1. Load data
        # ========================================
        excel_path = self._resolve_excel_path(self.FILE_PATH)
        df_raw = pd.read_excel(excel_path, sheet_name=self.SHEET_NAME, header=6)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        
        # ========================================
        # 2. Extract required columns
        # ========================================
        height_col = self.HEIGHT_COL.strip()
        shoulder_col = self.SHOULDER_COL.strip()
        waist_col = self.WAIST_COL.strip()
        hip_col = self.HIP_COL.strip()
        knee_col = self.KNEE_COL.strip()
        ankle_col = self.ANKLE_COL.strip()
        foot_col = self.FOOT_COL.strip()

        use_cols = [
            height_col, shoulder_col, waist_col, hip_col,
            knee_col, ankle_col, foot_col
        ]
        missing = [c for c in use_cols if c not in df_raw.columns]
        if missing:
            available_preview = ", ".join(df_raw.columns[:20])
            raise KeyError(
                f"Missing columns: {missing}. Available (first 20): {available_preview}"
            )

        df = df_raw[use_cols].dropna()
        
        # ========================================
        # 3. Calculate segment lengths
        #    trunk = shoulder height - hip height
        #    thigh = hip height - knee height
        #    shank = knee height - ankle height
        #    foot  = foot length
        # ========================================
        df["trunk_len"] = df[shoulder_col] - df[hip_col]
        df["thigh_len"] = df[hip_col] - df[knee_col]
        df["shank_len"] = df[knee_col] - df[ankle_col]
        df["foot_len"] = df[foot_col]
        
        # Remove outliers where length is negative or zero
        df = df[
            (df["trunk_len"] > 0) &
            (df["thigh_len"] > 0) &
            (df["shank_len"] > 0) &
            (df[height_col] > 0)
        ].copy()
        
        # ========================================
        # 4. Normalize to ratios (segment / height)
        # ========================================
        df["trunk_ratio"] = df["trunk_len"] / df[height_col]
        df["thigh_ratio"] = df["thigh_len"] / df[height_col]
        df["shank_ratio"] = df["shank_len"] / df[height_col]
        df["foot_ratio"] = df["foot_len"] / df[height_col]
        
        self.df = df

    @staticmethod
    def _resolve_excel_path(path_value: str) -> str:
        """Resolve Excel file path relative to this script if needed."""
        candidate = Path(path_value)
        if candidate.is_file():
            return str(candidate)

        script_dir = Path(__file__).resolve().parent
        candidate = script_dir / path_value
        if candidate.is_file():
            return str(candidate)

        project_root = script_dir.parents[3] if len(script_dir.parents) >= 4 else script_dir
        candidate = project_root / path_value
        if candidate.is_file():
            return str(candidate)

        raise FileNotFoundError(f"Excel file not found: {path_value}")
    
    def _train_models(self):
        """Train linear regression models for each segment ratio."""
        # Regression input: height + foot length
        height_col = self.HEIGHT_COL.strip()
        foot_col = self.FOOT_COL.strip()
        X = self.df[[height_col, foot_col]].values
        
        # ========================================
        # 5. Train linear regression models for each segment ratio
        #    (height, foot_length) -> each ratio
        # ========================================
        targets = ["trunk_ratio", "thigh_ratio", "shank_ratio", "foot_ratio"]
        
        for t in targets:
            model = LinearRegression()
            model.fit(X, self.df[t].values)
            self.models[t] = model
    
    def predict_segment_ratios(self, height: float, foot_length: float) -> Dict[str, float]:
        """
        Predict average body segment ratios based on 1st-2nd year direct measurement data.
        Estimates segment length ratios from height and foot length.

        Parameters
        ----------
        height : float
            Height (in mm)
        foot_length : float
            Foot length (in mm, based on measurement 122)

        Returns
        -------
        Dict[str, float]
            {
              'trunk_ratio': ...,
              'thigh_ratio': ...,
              'shank_ratio': ...,
              'foot_ratio': ...
            }
        """
        X_input = np.array([[height, foot_length]])

        return {
            "trunk_ratio": round(float(self.models["trunk_ratio"].predict(X_input)[0]), 3),
            "thigh_ratio": round(float(self.models["thigh_ratio"].predict(X_input)[0]), 3),
            "shank_ratio": round(float(self.models["shank_ratio"].predict(X_input)[0]), 3),
            "foot_ratio": round(float(self.models["foot_ratio"].predict(X_input)[0]), 3),
        }
    
    def predict_segment_lengths(self, height: float, foot_length: float) -> Dict[str, float]:
        """
        Return both segment ratios and actual lengths (in mm).
        
        Parameters
        ----------
        height : float
            Height (in mm)
        foot_length : float
            Foot length (in mm)
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing segment lengths (mm) and ratios
        """
        ratios = self.predict_segment_ratios(height, foot_length)
        return {
            "trunk_len_mm": round(height * ratios["trunk_ratio"], 1),
            "thigh_len_mm": round(height * ratios["thigh_ratio"], 1),
            "shank_len_mm": round(height * ratios["shank_ratio"], 1),
            "foot_len_mm": round(height * ratios["foot_ratio"], 1),
            **ratios,
        }


# ========================================
# Test Code
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Body Segment Ratio Predictor - Test")
    print("=" * 60)
    
    try:
        # Initialize predictor
        print("\n[1] Initializing predictor...")
        file_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
        predictor = BodySegmentRatioPredictor(file_path=file_path_arg)
        print(f"✓ Model loaded successfully")
        print(f"✓ Training data shape: {predictor.df.shape}")
        print(f"✓ Number of samples: {len(predictor.df)}")
        
        # Display model coefficients
        print("\n[2] Model Coefficients (Height, Foot Length):")
        for segment, model in predictor.models.items():
            coef = model.coef_
            intercept = model.intercept_
            print(f"  {segment:15s}: coef={coef}, intercept={intercept:.6f}")
        
        # Test case 1: Average Korean adult male
        print("\n[3] Test Case 1: Average Korean Adult Male")
        height_1 = 1730  # mm (173 cm)
        foot_1 = 255     # mm (25.5 cm)
        print(f"  Input: height={height_1}mm, foot_length={foot_1}mm")
        
        ratios_1 = predictor.predict_segment_ratios(height_1, foot_1)
        print(f"  Ratios: {ratios_1}")
        
        lengths_1 = predictor.predict_segment_lengths(height_1, foot_1)
        print(f"  Lengths: {lengths_1}")
        
        # Test case 2: Smaller person
        print("\n[4] Test Case 2: Smaller Person")
        height_2 = 1600  # mm (160 cm)
        foot_2 = 235     # mm (23.5 cm)
        print(f"  Input: height={height_2}mm, foot_length={foot_2}mm")
        
        ratios_2 = predictor.predict_segment_ratios(height_2, foot_2)
        print(f"  Ratios: {ratios_2}")
        
        lengths_2 = predictor.predict_segment_lengths(height_2, foot_2)
        print(f"  Lengths: {lengths_2}")
        
        # Test case 3: Taller person
        print("\n[5] Test Case 3: Taller Person")
        height_3 = 1850  # mm (185 cm)
        foot_3 = 280     # mm (28 cm)
        print(f"  Input: height={height_3}mm, foot_length={foot_3}mm")
        
        ratios_3 = predictor.predict_segment_ratios(height_3, foot_3)
        print(f"  Ratios: {ratios_3}")
        
        lengths_3 = predictor.predict_segment_lengths(height_3, foot_3)
        print(f"  Lengths: {lengths_3}")
        
        # Validation: Check if sum of ratios is reasonable
        print("\n[6] Validation:")
        for i, (height, foot, ratios) in enumerate([
            (height_1, foot_1, ratios_1),
            (height_2, foot_2, ratios_2),
            (height_3, foot_3, ratios_3)
        ], 1):
            total_ratio = sum(ratios.values())
            print(f"  Test Case {i}: Sum of ratios = {total_ratio:.3f}")
            if 0.5 < total_ratio < 1.5:
                print(f"    ✓ Ratios are within reasonable range")
            else:
                print(f"    ⚠ Warning: Ratios may be out of expected range")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Data file not found")
        print(f"  Please ensure '{BodySegmentRatioPredictor.FILE_PATH}' exists")
        print(f"  Error details: {e}")
    except Exception as e:
        print(f"\n✗ Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
