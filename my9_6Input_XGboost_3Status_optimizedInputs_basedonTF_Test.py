import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ===== Configuration =====
MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]

DATA_FOLDER_L = 'C:/pythonFiles/inputs'  #     'G:/3-ALL Python and AI/my codes/inputs'
DATA_FOLDER = 'C:/Users/Hossein/AppData/Roaming/MetaQuotes/Terminal/Common/Files/'   # 'G:/3-ALL Python and AI/my codes/input_Buy_file'

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

# ===== Thresholds for categorization =====
LOWER_THRESHOLD = -0.3
UPPER_THRESHOLD = 0.3

# ===== Data filtering and categorization function =====
def filter_and_categorize(df):
    def categorize_profit(val):
        if val > UPPER_THRESHOLD:
            return 1  # Profit
        elif val < LOWER_THRESHOLD:
            return 0  # Loss
        else:
            return -1  # Filter out

    df['target'] = df['Sood_zarar'].apply(categorize_profit)
    print(f"Samples before filtering: {len(df)}")
    df_filtered = df[df['target'] != -1].reset_index(drop=True)
    print(f"Samples after filtering: {len(df_filtered)}")
    return df_filtered

def analyze_profit_distribution(file_path, model_tag):
    data = pd.read_csv(file_path)
    data = filter_and_categorize(data)

    allowed_timeframes_map = {
        '15minBuy': ['15 min='],
        '1hBuy': ['1h=', '4h=', 'D='],
        '2hBuy': ['4h=', 'D='],
        '3hBuy': ['4h=', 'D='],
        '4hBuy': ['4h=', 'D='],
        '1DBuy': ['D=']
    }
    allowed_timeframes = allowed_timeframes_map.get(model_tag, [])

    def is_allowed_col(col_name):
        if col_name in ['Sood_zarar', 'target']:
            return False
        if col_name in EXCLUDE_COLUMNS:
            return False
        return any(tf in col_name for tf in allowed_timeframes)

    filtered_cols = [col for col in data.columns if is_allowed_col(col)]
    filtered_data = data[filtered_cols + ['target']]

    total = len(filtered_data)
    counts = filtered_data['target'].value_counts().to_dict()

    profit = counts.get(1, 0)
    loss = counts.get(0, 0)

    print(f"{model_tag} data distribution (total={total}):")
    print(f"  Profit: {profit} ({profit / total * 100:.2f}%)")
    print(f"  Loss: {loss} ({loss / total * 100:.2f}%)\n")

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
        print(f'Max depth={self.model.get_params()["max_depth"]} Test accuracy: {accuracy:.4f}')
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
        values = [-1, 1]  # Loss, Profit
        continuous_value = sum(p * v for p, v in zip(probs, values))
        return continuous_value

    def predict(self, data_dict):
        X = self.preprocess_new_data(data_dict)
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]

        continuous_pred = self.convert_probs_to_continuous(pred_proba)

        labels = {0: 'Loss', 1: 'Profit'}
        pred_label = labels[pred_class]

        return pred_class, continuous_pred, pred_label, pred_proba

def read_input_file(filename):
    df = pd.read_csv(filename)
    return df.iloc[0].to_dict()

if __name__ == '__main__':
    print("Loading and training models...")

    predictors = {}

    for tag in MODEL_TAGS:
        print(f"Training model: {tag}")
        loader = DataLoaderAndPreprocessor(TRAIN_PATHS[tag], tag)
        X, y = loader.load_and_preprocess()

        analyze_profit_distribution(TRAIN_PATHS[tag], tag)

        trainer = ModelTrainer()
        trainer.train(X, y)

        model = trainer.get_model()
        predictor = Predictor(model, loader.label_encoders, loader.cat_columns, list(X.columns))
        predictors[tag] = predictor

    print("All models trained. Waiting for input...")

    while True:
        if os.path.exists(INPUT_PATH):
            try:
                input_data = read_input_file(INPUT_PATH)

                for tag, predictor in predictors.items():
                    pred_class, continuous_pred, pred_label, pred_proba = predictor.predict(input_data)
                    output_path = OUTPUT_PATHS[tag]

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"{pred_proba[1]:.6f} {pred_proba[0]:.6f}")

                    print(f"{tag} prediction: {pred_label} | Probabilities: Profit={pred_proba[1]:.3f}, Loss={pred_proba[0]:.3f}")

                os.remove(INPUT_PATH)
                print("Prediction done. Waiting for next input...")

            except Exception as e:
                print("Error processing input:", e)

        time.sleep(1)
