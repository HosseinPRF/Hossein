import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ===== پیکربندی =====
MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]

DATA_FOLDER_L = 'C:/pythonFiles/inputs'  
DATA_FOLDER = 'C:/Users/Hossein/AppData/Roaming/MetaQuotes/Terminal/Common/Files/'
#'C:/Users/Hossein/AppData/Roaming/MetaQuotes/Tester/D0E8209F77C8CF37AD8BF550E51FF075/Agent-127.0.0.1-3000/MQL5/Files'

INPUT_FILE = 'inputFile_Python2.csv'
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


# ===================== Main Loop =====================

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

    print("\n✅ All models trained. Waiting for input...\n")

    while True:
        if os.path.exists(INPUT_PATH):
            try:
                input_data = read_input_file(INPUT_PATH)

                for tag, predictor in predictors.items():
                    pred_class, continuous_pred, pred_label, pred_proba = predictor.predict(input_data)
                    output_path = OUTPUT_PATHS[tag]

                    # فقط نوشتن احتمال ها به ترتیب: سودده، بدون تغییر، ضررده
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"{pred_proba[2]:.6f} {pred_proba[1]:.6f} {pred_proba[0]:.6f}")

                    print(f"📥 {tag} → Prediction: {pred_label} | "
                          f"Probs: سودده={pred_proba[2]:.3f}, بدون تغییر={pred_proba[1]:.3f}, ضررده={pred_proba[0]:.3f}")

                os.remove(INPUT_PATH)
                print("\n✅ Predictions done. Waiting for next input...\n")
            except Exception as e:
                print("❌ Error while processing input:", e)

        time.sleep(1)
