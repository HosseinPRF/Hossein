import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from pandastable import Table
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ===== Configuration =====
MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]

DATA_FOLDER_L = 'C:/pythonFiles/inputs/CSVBuy'
DATA_FOLDER = 'C:/Users/Hossein/AppData/Roaming/MetaQuotes/Terminal/Common/Files/'

INPUT_FILE = 'inputFile_Python2.csv'
INPUT_PATH = os.path.join(DATA_FOLDER, INPUT_FILE)

TRAIN_FILES = {tag: f"{tag}.csv" for tag in MODEL_TAGS}
TRAIN_PATHS = {tag: os.path.join(DATA_FOLDER_L, fname) for tag, fname in TRAIN_FILES.items()}

OUTPUT_FILES = {tag: f"prediction_{tag}.txt" for tag in MODEL_TAGS}
OUTPUT_PATHS = {tag: os.path.join(DATA_FOLDER, fname) for tag, fname in OUTPUT_FILES.items()}

EXCLUDE_COLUMNS = []

LOWER_THRESHOLD = -0.15
UPPER_THRESHOLD = 0.15

# ===== Utility Functions =====
def filter_and_categorize(df):
    def categorize_profit(val):
        if val > UPPER_THRESHOLD:
            return 1
        elif val < LOWER_THRESHOLD:
            return 0
        else:
            return -1
    df['target'] = df['Sood_zarar'].apply(categorize_profit)
    return df[df['target'] != -1].reset_index(drop=True)

def plot_feature_importance(model, X):
    importance = model.get_booster().get_score(importance_type='weight')
    importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(10, 8))
    plt.barh(list(importance.keys()), list(importance.values()))
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

def check_feature_correlation(X, threshold=0.9):
    corr_matrix = X.corr(numeric_only=True)
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_features.add(corr_matrix.columns[i])
    return X.drop(columns=correlated_features), correlated_features

# ===== Classes =====
class DataLoaderAndPreprocessor:
    def __init__(self, file_path, model_tag):
        self.file_path = file_path
        self.model_tag = model_tag
        self.label_encoders = {}
        self.cat_columns = []
        self.X = None
        self.y = None
        self.filtered_data = None

    def load_and_preprocess(self):
        data = pd.read_csv(self.file_path, low_memory=False)
        data = filter_and_categorize(data)
        self.filtered_data = data.copy()

        allowed_timeframes_map = {
            '15minBuy': ['15 min='],
            '1hBuy': ['1h=', '4h=', 'D='],
            '2hBuy': ['4h=', 'D='],
            '3hBuy': ['4h=', 'D='],
            '4hBuy': ['4h=', 'D='],
            '1DBuy': ['D=']
        }
        allowed_timeframes = allowed_timeframes_map.get(self.model_tag, [])

        def is_valid(col):
            if col in ['Sood_zarar', 'target'] or col in EXCLUDE_COLUMNS:
                return False
            return any(tf in col for tf in allowed_timeframes)

        selected_cols = [col for col in data.columns if is_valid(col)]
        features = data[selected_cols].copy()

        self.cat_columns = []

        for col in features.columns:
            if features[col].dtype == 'object' or features[col].map(type).nunique() > 1:
                features[col] = features[col].astype(str)
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col])
                self.label_encoders[col] = le
                self.cat_columns.append(col)

        self.X = features
        self.y = data['target']
        return self.X, self.y

class ModelTrainer:
    def __init__(self, file_path, filtered_data, max_depth=5):
        self.file_path = file_path
        self.filtered_data = filtered_data
        self.model = XGBClassifier(max_depth=max_depth, random_state=42, eval_metric='logloss')
        self.selected_columns = []

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, dropped_features = check_feature_correlation(X_train)
        self.selected_columns = X_train.columns.tolist()
        X_test = X_test[self.selected_columns]

        self.model.fit(X_train, y_train)

        print("Correlation matrix:\n", X_train.corr())
        plot_feature_importance(self.model, X_train)

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")

        test_indices = X_test.index.tolist()
        display_df = X_test.copy()
        display_df['Sood_zarar'] = self.filtered_data.loc[test_indices, 'Sood_zarar'].values
        display_df['Actual'] = y_test.values
        display_df['Predicted'] = y_pred
        display_df['Profit_Proba'] = y_prob[:, 1]
        display_df['Loss_Proba'] = y_prob[:, 0]

        root = tk.Tk()
        root.title("Test Results - All Features + Sood_zarar")
        frame = tk.Frame(root)
        frame.pack(fill='both', expand=True)
        table = Table(frame, dataframe=display_df)
        table.show()
        root.mainloop()

    def get_model(self):
        return self.model

class Predictor:
    def __init__(self, model, label_encoders, cat_columns, feature_columns):
        self.model = model
        self.label_encoders = label_encoders
        self.cat_columns = cat_columns
        self.feature_columns = feature_columns

    def preprocess(self, row_dict):
        df = pd.DataFrame([row_dict])
        df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns], inplace=True, errors='ignore')

        for col in self.cat_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                le = self.label_encoders.get(col)
                if le:
                    if df.at[0, col] in le.classes_:
                        df[col] = le.transform([df.at[0, col]])[0]
                    else:
                        df[col] = -1
                else:
                    df[col] = 0

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_columns]

    def convert_probs(self, probs):
        return sum(p * v for p, v in zip(probs, [-1, 1]))

    def predict(self, row_dict):
        X = self.preprocess(row_dict)
        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]
        return pred, self.convert_probs(proba), {0: 'Loss', 1: 'Profit'}[pred], proba

def read_input_file(path):
    df = pd.read_csv(path)
    return df.to_dict(orient='records')

# ===== Main Execution =====
if __name__ == '__main__':
    print("Loading and training models...")

    predictors = {}

    for tag in MODEL_TAGS:
        print(f"\n>>> Training model: {tag}")
        loader = DataLoaderAndPreprocessor(TRAIN_PATHS[tag], tag)
        X, y = loader.load_and_preprocess()
        trainer = ModelTrainer(TRAIN_PATHS[tag], loader.filtered_data)
        trainer.train(X, y)
        model = trainer.get_model()
        predictor = Predictor(model, loader.label_encoders, loader.cat_columns, trainer.selected_columns)
        predictors[tag] = predictor

    print("\n✅ All models trained. Waiting for input...")

    while True:
        if os.path.exists(INPUT_PATH):
            try:
                all_inputs = read_input_file(INPUT_PATH)
                for row in all_inputs:
                    for tag, predictor in predictors.items():
                        pred_class, cont_pred, label, probs = predictor.predict(row)
                        output_path = OUTPUT_PATHS[tag]
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(f"{probs[1]:.6f} {probs[0]:.6f}")
                        print(f"[{tag}] Prediction: {label} | Profit={probs[1]:.3f}, Loss={probs[0]:.3f}")
                os.remove(INPUT_PATH)
                print("✅ Prediction done. Waiting for next input...")
            except Exception as e:
                print("❌ Error processing input:", e)

        time.sleep(1)
