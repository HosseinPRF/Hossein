import os
import time
import pandas as pd
from tkinter import Tk, Frame, Label, Listbox, Scrollbar, Button, RIGHT, LEFT, Y, BOTH, END
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ==== پیکربندی و کدهای بارگذاری و آموزش مدل همانند قبل ====

MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]
DATA_FOLDER_L = 'G:/3-ALL Python and AI/my codes/inputs'
DATA_FOLDER = 'G:/3-ALL Python and AI/my codes/input_Buy_file'
TRAIN_FILES = {tag: f"{tag}.csv" for tag in MODEL_TAGS}
TRAIN_PATHS = {tag: os.path.join(DATA_FOLDER_L, fname) for tag, fname in TRAIN_FILES.items()}

EXCLUDE_COLUMNS = [
    'Supp_Z_TF 15 min=', 'Supp_Z_TF 1h=', 'Supp_Z_TF 4h=', 'Supp_Z_TF D=',
    'Ress_Z_TF 15 min=', 'Ress_Z_TF 1h=', 'Ress_Z_TF 4h=', 'Ress_Z_TF D=',
    'n_bar_Change_ravand_TF 15 min=', 'n_bar_Change_ravand_TF 1h=', 'n_bar_Change_ravand_TF 4h=', 'n_bar_Change_ravand_TF D=',
    'n_bar_Change_ravand_Zstr 15 min=', 'n_bar_Change_ravand_Zstr 1h=', 'n_bar_Change_ravand_Zstr 4h=', 'n_bar_Change_ravand_Zstr D=',
    'now_ravand_perc_TF 15 min=', 'now_ravand_perc_TF 1h=', 'now_ravand_perc_TF 4h=', 'now_ravand_perc_TF D=',
    'sec_ravand_perc_TF 15 min=', 'sec_ravand_perc_TF 1h=', 'sec_ravand_perc_TF 4h=', 'sec_ravand_perc_TF D=',
    'third_ravand_perc_TF 15 min=', 'third_ravand_perc_TF 1h=', 'third_ravand_perc_TF 4h=', 'third_ravand_perc_TF D='
]

LOWER_THRESHOLD = -0.3
UPPER_THRESHOLD = 0.3

def filter_and_categorize(df):
    def categorize_profit(val):
        if val > UPPER_THRESHOLD:
            return 1
        elif val < LOWER_THRESHOLD:
            return 0
        else:
            return -1
    df['target'] = df['Sood_zarar'].apply(categorize_profit)
    df_filtered = df[df['target'] != -1].reset_index(drop=True)
    return df_filtered

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
        data = filter_and_categorize(data)

        if self.model_tag == '15minBuy':
            allowed_timeframes = ['15 min=']
        else:
            allowed_timeframes_map = {
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
            return any(tf in col_name for tf in allowed_timeframes)

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
        self.model = XGBClassifier(max_depth=max_depth, random_state=random_state, eval_metric='logloss')

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy on test set: {accuracy:.4f}')
        return accuracy

    def get_model(self):
        return self.model

# ===== GUI Part =====

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FeatureImportanceGUI(tk.Tk):
    def __init__(self, models, feature_names):
        super().__init__()
        self.title("Feature Importance Viewer")
        self.geometry("900x600")

        self.models = models  # dict: tag -> model
        self.feature_names = feature_names  # dict: tag -> list of features

        # Left frame: listbox for model selection
        frame_left = ttk.Frame(self)
        frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        label = ttk.Label(frame_left, text="Select Model:")
        label.pack()

        self.listbox = tk.Listbox(frame_left, height=10)
        for tag in self.models.keys():
            self.listbox.insert(tk.END, tag)
        self.listbox.pack(fill=tk.Y)
        self.listbox.bind('<<ListboxSelect>>', self.on_model_select)

        # Right frame: matplotlib plot area
        frame_right = ttk.Frame(self)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(7,6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.selected_model_tag = None

    def on_model_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            tag = event.widget.get(index)
            self.selected_model_tag = tag
            self.plot_feature_importance(tag)

    def plot_feature_importance(self, tag):
        model = self.models[tag]
        features = self.feature_names[tag]

        self.ax.clear()

        # استخراج اهمیت ویژگی ها
        importance = model.feature_importances_
        # مرتب سازی به ترتیب نزولی
        sorted_idx = importance.argsort()[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]

        # رسم نمودار
        self.ax.barh(sorted_features[:20][::-1], sorted_importance[:20][::-1])
        self.ax.set_title(f"Feature Importance for {tag}")
        self.ax.set_xlabel("Importance")
        self.ax.set_ylabel("Feature")
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    print("Training models and preparing GUI...")

    predictors = {}
    feature_names = {}
    models = {}

    for tag in MODEL_TAGS:
        print(f"Training model: {tag}")
        loader = DataLoaderAndPreprocessor(TRAIN_PATHS[tag], tag)
        X, y = loader.load_and_preprocess()

        trainer = ModelTrainer()
        acc = trainer.train(X, y)
        model = trainer.get_model()

        models[tag] = model
        feature_names[tag] = list(X.columns)

    app = FeatureImportanceGUI(models, feature_names)
    app.mainloop()

if __name__ == "__main__":
    main()
