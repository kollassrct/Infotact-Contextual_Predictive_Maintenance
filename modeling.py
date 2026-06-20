import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

from src.utils import NoiseAugmentor

class MaintenanceModel:
    def __init__(self, data_path='/home/kartik/.gemini/antigravity/scratch/predictive_maintenance/data/fused_dataset.csv'):
        self.data_path = data_path
        self.model_dir = '/home/kartik/.gemini/antigravity/scratch/predictive_maintenance/models'
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Fused data not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        drop_cols = ['udi', 'product_id', 'failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df['failure']
        
        X = pd.get_dummies(X, columns=['type'], drop_first=True)
        return X, y

    def train_with_cv(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        f1_scores = []
        best_model = None
        best_f1 = 0
        
        print("Starting 5-fold Stratified CV with Noise Augmentation & SMOTE...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline = ImbPipeline([
                ('noise', NoiseAugmentor(noise_std=0.03)), # Inject noise during training
                ('smote', SMOTE(random_state=42)),
                ('classifier', lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=31,
                    importance_type='gain',
                    random_state=42,
                    verbose=-1
                ))
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            fold_f1 = f1_score(y_val, y_pred, average='macro')
            f1_scores.append(fold_f1)
            
            print(f"Fold {fold+1} Macro F1: {fold_f1:.4f}")
            
            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_model = pipeline
                
        avg_f1 = np.mean(f1_scores)
        print(f"\nAverage Macro F1: {avg_f1:.4f}")
        
        # Save the best model
        model_path = os.path.join(self.model_dir, 'lgbm_smote_model.joblib')
        joblib.dump(best_model, model_path)
        print(f"Best model saved to {model_path}")
        
        return best_model, avg_f1

if __name__ == "__main__":
    trainer = MaintenanceModel()
    try:
        X, y = trainer.prepare_data()
        print(f"Feature set shape: {X.shape}")
        model, avg_f1 = trainer.train_with_cv(X, y)
        
        if avg_f1 >= 0.85:
            print("SUCCESS: Target Macro F1 reached!")
        else:
            print("WARNING: Macro F1 below target. Consider further tuning.")
            
    except Exception as e:
        print(f"Error during modeling: {e}")
