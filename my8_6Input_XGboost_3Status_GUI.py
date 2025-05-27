import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import tkinter as tk
from tkinter import ttk, messagebox

# ===== پیکربندی =====
MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]

DATA_FOLDER_L = 'G:/3-ALL Python and AI/my codes/inputs'  # 'C:/pythonFiles'
DATA_FOLDER = 'C:/Users/Hossein/AppData/Roaming/MetaQuotes/Tester/D0E8209F77C8CF37AD8BF550E51FF075/Agent-127.0.0.1-3000/MQL5/Files'

INPUT_FILE = 'inputFile_Python.csv'
INPUT_PATH = os.path.join(DATA_FOLDER, INPUT_FILE)

TRAIN_FILES = {tag: f"{tag}.csv" for tag in MODEL_TAGS}
TRAIN_PATHS = {tag: os.path.join(DATA_FOLDER_L, fname) for tag, fname in TRAIN_FILES.items()}

OUTPUT_FILES = {tag: f"prediction_{tag}.txt" for tag in MODEL_TAGS}
OUTPUT_PATHS = {tag: os.path.join(DATA_FOLDER, fname) for tag, fname in OUTPUT_FILES.items()}


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

    def predict(self, data_dict):
        X = self.preprocess_new_data(data_dict)
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0][pred_class]
        labels = {0: 'ضررده', 1: 'بدون تغییر', 2: 'سودده'}
        return pred_class, pred_proba, labels[pred_class]


def read_input_file(filename):
    df = pd.read_csv(filename)
    return df.iloc[0].to_dict()


# ===================== Training models =====================

print("🔄 Loading and training models...")

predictors = {}
models_feature_importance = {}  # ذخیره feature importance هر مدل

for tag in MODEL_TAGS:
    print(f"\n🧠 Training model for {tag} prediction...")
    loader = DataLoaderAndPreprocessor(TRAIN_PATHS[tag])
    X, y = loader.load_and_preprocess()

    trainer = ModelTrainer()
    trainer.train(X, y)

    model = trainer.get_model()
    predictor = Predictor(model, loader.label_encoders, loader.cat_columns, list(X.columns))
    predictors[tag] = predictor

    # ذخیره اهمیت ویژگی‌ها
    importance_scores = model.feature_importances_
    feature_names = list(X.columns)
    models_feature_importance[tag] = list(zip(feature_names, importance_scores))

print("\n✅ All models trained. Starting GUI...\n")

# ===================== GUI =====================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("مدل پیش‌بینی با XGBoost و نمایش اثر پارامترها")
        self.root.geometry("600x500")

        # انتخاب مدل
        tk.Label(root, text="انتخاب مدل:").pack(pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(root, textvariable=self.model_var, values=MODEL_TAGS, state="readonly")
        self.model_combo.pack(pady=5)
        self.model_combo.current(0)

        # دکمه نمایش اثر ویژگی‌ها
        self.show_button = tk.Button(root, text="نمایش اهمیت ویژگی‌ها", command=self.show_feature_importance)
        self.show_button.pack(pady=10)

        # جعبه متن برای نمایش نتایج
        self.text_box = tk.Text(root, height=20, width=70)
        self.text_box.pack(padx=10, pady=10)

        # دکمه خروج
        self.quit_button = tk.Button(root, text="خروج", command=root.quit)
        self.quit_button.pack(pady=10)

    def show_feature_importance(self):
        selected_model = self.model_var.get()
        self.text_box.delete(1.0, tk.END)
        if selected_model not in models_feature_importance:
            self.text_box.insert(tk.END, "مدل انتخاب شده موجود نیست.")
            return

        features = models_feature_importance[selected_model]
        features_sorted = sorted(features, key=lambda x: x[1], reverse=True)

        self.text_box.insert(tk.END, f"اهمیت ویژگی‌ها برای مدل {selected_model}:\n\n")
        for feat, score in features_sorted:
            self.text_box.insert(tk.END, f"{feat}: {score:.4f}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
