import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk
import threading

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
        data['target'] = (data['Sood_zarar'] > 0).astype(int)
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
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

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
        pred_proba = self.model.predict_proba(X)[0][1]
        return pred_class, pred_proba


def read_input_file(filename):
    df = pd.read_csv(filename)
    return df.iloc[0].to_dict()


# ===================== GUI Class =====================

class FeatureImportanceGUI:
    def __init__(self, master, predictors):
        self.master = master
        self.predictors = predictors

        master.title("Feature Importance Viewer")
        master.geometry("600x400")

        self.label = tk.Label(master, text="Select a model to view feature importances:")
        self.label.pack(pady=10)

        # لیست مدل‌ها
        self.model_listbox = tk.Listbox(master)
        for tag in self.predictors.keys():
            self.model_listbox.insert(tk.END, tag)
        self.model_listbox.pack(pady=10)
        self.model_listbox.bind("<<ListboxSelect>>", self.show_importances)

        # جدول نمایش ویژگی‌ها و میزان اهمیت‌شون
        self.tree = ttk.Treeview(master, columns=("Feature", "Importance"), show="headings")
        self.tree.heading("Feature", text="Feature")
        self.tree.heading("Importance", text="Importance")
        self.tree.pack(expand=True, fill=tk.BOTH, pady=10)

    def show_importances(self, event):
        selected_idx = self.model_listbox.curselection()
        if not selected_idx:
            return

        tag = self.model_listbox.get(selected_idx)
        predictor = self.predictors[tag]
        model = predictor.model
        features = predictor.feature_columns

        importances = model.feature_importances_

        # پاک کردن محتویات قبلی جدول
        for i in self.tree.get_children():
            self.tree.delete(i)

        # مرتب کردن بر اساس اهمیت به صورت نزولی
        sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

        for feature, importance in sorted_features:
            self.tree.insert("", tk.END, values=(feature, f"{importance:.4f}"))


# ===================== Main Loop =====================

def main():
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

    # اجرای GUI
    root = tk.Tk()
    app = FeatureImportanceGUI(root, predictors)
    root.mainloop()


if __name__ == '__main__':
    main()
