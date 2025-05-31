import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# ===== پیکربندی =====
MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]

DATA_FOLDER_L = 'G:/3-ALL Python and AI/my codes/inputs'  # مسیر فایل‌های آموزش
DATA_FOLDER = 'G:/3-ALL Python and AI/my codes/input_Buy_file'  # مسیر فایل‌های ورودی و خروجی

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
    def __init__(self, file_path, model_tag):
        self.file_path = file_path
        self.model_tag = model_tag
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

        allowed_timeframes_map = {
            '15minBuy': ['15 min=', '1h=', '2h=', '3h=', '4h=', 'D='],
            '1hBuy': ['1h=', '4h=', 'D='],
            '2hBuy': ['4h=', 'D='],
            '3hBuy': ['4h=', 'D='],
            '4hBuy': ['4h=', 'D='],
            '1DBuy': ['D=']
        }
        allowed_timeframes = allowed_timeframes_map.get(self.model_tag, [])

        def is_allowed_col(col_name):
            if col_name in ['Sood_zarar', 'target']:
                return False
            if col_name in EXCLUDE_COLUMNS:
                return False
            for tf in allowed_timeframes:
                if tf in col_name:
                    return True
            return False

        filtered_cols = [col for col in data.columns if is_allowed_col(col)]

        data_features = data[filtered_cols]

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
        values = [-1, 0, 1]
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


# ===== GUI =====
class FeatureImportanceApp:
    def __init__(self, root, predictors):
        self.root = root
        self.root.title("نمایش تاثیر پارامترها در مدل‌ها")
        self.predictors = predictors

        self.model_var = tk.StringVar()
        self.model_var.set(list(predictors.keys())[0])

        ttk.Label(root, text="انتخاب مدل:").pack(pady=5)
        self.model_menu = ttk.OptionMenu(root, self.model_var, self.model_var.get(), *predictors.keys(), command=self.update_plot)
        self.model_menu.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.update_plot(self.model_var.get())

    def update_plot(self, selected_model):
        self.ax.clear()
        predictor = self.predictors[selected_model]
        model = predictor.model

        importances = model.feature_importances_
        features = predictor.feature_columns

        sorted_idx = importances.argsort()[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_features = [features[i] for i in sorted_idx]

        self.ax.barh(sorted_features[:20], sorted_importances[:20])
        self.ax.set_xlabel("اهمیت ویژگی")
        self.ax.set_title(f"Feature Importance مدل {selected_model}")
        self.ax.invert_yaxis()
        self.fig.tight_layout()

        self.canvas.draw()


def train_models():
    predictors = {}
    for tag in MODEL_TAGS:
        print(f"\n🧠 آموزش مدل برای {tag} ...")
        loader = DataLoaderAndPreprocessor(TRAIN_PATHS[tag], tag)
        X, y = loader.load_and_preprocess()

        trainer = ModelTrainer()
        trainer.train(X, y)

        model = trainer.get_model()
        predictor = Predictor(model, loader.label_encoders, loader.cat_columns, list(X.columns))
        predictors[tag] = predictor
    return predictors


if __name__ == '__main__':
    # ابتدا مدل‌ها را آموزش می‌دهیم
    predictors = train_models()

    # سپس GUI را باز می‌کنیم
    root = tk.Tk()
    app = FeatureImportanceApp(root, predictors)
    root.mainloop()
