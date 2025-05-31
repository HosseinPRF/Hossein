import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ===== پیکربندی =====
MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]

DATA_FOLDER_L = 'G:/3-ALL Python and AI/my codes/inputs'  #   'C:/pythonFiles/inputs'
DATA_FOLDER = 'G:/3-ALL Python and AI/my codes/input_Buy_file' # 'C:/Users/Hossein/AppData/Roaming/MetaQuotes/Terminal/Common/Files/'

INPUT_FILE = 'inputFile_Python2.csv'
INPUT_PATH = os.path.join(DATA_FOLDER, INPUT_FILE)

TRAIN_FILES = {tag: f"{tag}.csv" for tag in MODEL_TAGS}
TRAIN_PATHS = {tag: os.path.join(DATA_FOLDER_L, fname) for tag, fname in TRAIN_FILES.items()}

OUTPUT_FILES = {tag: f"prediction_{tag}.txt" for tag in MODEL_TAGS}
OUTPUT_PATHS = {tag: os.path.join(DATA_FOLDER, fname) for tag, fname in OUTPUT_FILES.items()}

EXCLUDE_COLUMNS = [
    'Supp_Z_TF 15 min=', 'Supp_Z_TF 1h=', 'Supp_Z_TF 4h=', 'Supp_Z_TF D=',
    'Ress_Z_TF 15 min=', 'Ress_Z_TF 1h=', 'Ress_Z_TF 4h=', 'Ress_Z_TF D=',
    'n_bar_Change_ravand_TF 15 min=', 'n_bar_Change_ravand_TF 1h=', 'n_bar_Change_ravand_TF 4h=', 'n_bar_Change_ravand_TF D=',
    'n_bar_Change_ravand_Zstr 15 min=', 'n_bar_Change_ravand_Zstr 1h=', 'n_bar_Change_ravand_Zstr 4h=', 'n_bar_Change_ravand_Zstr D=',
    'now_ravand_perc_TF 15 min=', 'now_ravand_perc_TF 1h=', 'now_ravand_perc_TF 4h=', 'now_ravand_perc_TF D=',
    'sec_ravand_perc_TF 15 min=', 'sec_ravand_perc_TF 1h=', 'sec_ravand_perc_TF 4h=', 'sec_ravand_perc_TF D=',
    'third_ravand_perc_TF 15 min=', 'third_ravand_perc_TF 1h=', 'third_ravand_perc_TF 4h=', 'third_ravand_perc_TF D='
]

# ===================== Classes =====================

class DataLoaderAndPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.label_encoders = {}
        self.cat_columns = []
        self.X = None
        self.y = None

    def load_and_preprocess(self):
        data = pd.read_csv(self.file_path)

        def categorize_profit(val):
            if val < -1:
                return 0  # ضررده
            elif val > 1:
                return 2  # سودده
            else:
                return 1  # بدون تغییر

        data['target'] = data['Sood_zarar'].apply(categorize_profit)

        data_features = data.drop(columns=['Sood_zarar', 'target'])

        # حذف ستون‌های مشخص شده از داده‌های ویژگی
        data_features = data_features.drop(columns=[col for col in EXCLUDE_COLUMNS if col in data_features.columns])

        self.cat_columns = data_features.select_dtypes(include=['object']).columns.tolist()
        data_encoded = data_features.copy()

        for col in self.cat_columns:
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col])
            self.label_encoders[col] = le

        self.X = data_encoded
        self.y = data['target']
        return self.X, self.y


class ModelTrainer:
    def __init__(self, max_depth=5, random_state=42):
        self.model = XGBClassifier(max_depth=max_depth, random_state=random_state, eval_metric='mlogloss')

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'📊 Accuracy on test set: {accuracy:.4f}')
        self.X_train = X_train
        self.y_train = y_train

    def get_model(self):
        return self.model


class Predictor:
    def __init__(self, model, label_encoders, cat_columns, feature_columns):
        self.model = model
        self.label_encoders = label_encoders
        self.cat_columns = cat_columns
        self.feature_columns = feature_columns

    def preprocess_new_data(self, data_dict):
        new_df = pd.DataFrame([data_dict])

        # حذف ستون‌های مشخص شده در داده جدید
        new_df = new_df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in new_df.columns], errors='ignore')

        for col in self.cat_columns:
            if col in new_df.columns:
                val = new_df.at[0, col]
                le = self.label_encoders[col]
                if val in le.classes_:
                    new_df[col] = le.transform(new_df[col])
                else:
                    new_df[col] = -1
            else:
                new_df[col] = 0

        for col in self.feature_columns:
            if col not in new_df.columns:
                new_df[col] = 0

        new_df = new_df[self.feature_columns]
        return new_df

    def convert_probs_to_continuous(self, probs):
        values = [-1, 0, 1]  # ضررده، بدون تغییر، سودده
        continuous_value = sum(p * v for p, v in zip(probs, values))
        return continuous_value

    def predict(self, data_dict):
        X = self.preprocess_new_data(data_dict)
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]
        
        continuous_pred = self.convert_probs_to_continuous(pred_proba)
        
        labels = {0: 'ضررده', 1: 'بدون تغییر', 2: 'سودده'}
        pred_label = labels[pred_class]
        
        return pred_class, continuous_pred, pred_label, pred_proba


def read_input_file(filename):
    df = pd.read_csv(filename)
    return df.iloc[0].to_dict()

# ===================== GUI =====================

class FeatureImportanceGUI(tk.Tk):
    def __init__(self, predictors):
        super().__init__()
        self.title("نمایش اثر پارامترها روی مدل‌ها")
        self.geometry("900x600")

        self.predictors = predictors  # دیکشنری {tag: Predictor}

        self.create_widgets()

    def create_widgets(self):
        # لیست مدل‌ها
        label = tk.Label(self, text="انتخاب مدل:", font=("Tahoma", 12))
        label.pack(pady=5)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self, textvariable=self.model_var, values=list(self.predictors.keys()), state="readonly", font=("Tahoma", 11))
        self.model_combo.pack(pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

        # محلی برای رسم نمودار
        self.fig, self.ax = plt.subplots(figsize=(10,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_model_selected(self, event):
        tag = self.model_var.get()
        predictor = self.predictors[tag]
        model = predictor.model
        feature_names = predictor.feature_columns

        # استخراج اهمیت ویژگی از مدل xgboost
        try:
            importance = model.feature_importances_
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در استخراج اهمیت ویژگی‌ها:\n{e}")
            return

        # مرتب کردن بر اساس اهمیت
        sorted_idx = importance.argsort()
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]

        self.ax.clear()
        self.ax.barh(sorted_features, sorted_importance)
        self.ax.set_title(f"اهمیت ویژگی‌ها - مدل {tag}", fontsize=14)
        self.ax.set_xlabel("مقدار اهمیت")
        self.ax.set_ylabel("ویژگی‌ها")
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    print("🔄 Loading and training models...")

    predictors = {}

    for tag in MODEL_TAGS:
        print(f"\n🧠 Training model for {tag} prediction...")
        loader = DataLoaderAndPreprocessor(TRAIN_PATHS[tag])
        X, y = loader.load_and_preprocess()

        trainer = ModelTrainer()
        trainer.train(X, y)

        model = trainer.get_model()
        predictor = Predictor(model, loader.label_encoders, loader.cat_columns, list(X.columns))
        predictors[tag] = predictor

    print("\n✅ All models trained.")

    # اجرای GUI برای نمایش اهمیت ویژگی‌ها
    app = FeatureImportanceGUI(predictors)
    app.mainloop()
