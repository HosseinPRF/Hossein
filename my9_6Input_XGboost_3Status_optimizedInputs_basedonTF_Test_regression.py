import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from math import sqrt  # اضافه شده برای محاسبه RMSE

# ===== Configuration =====
MODEL_TAGS = ["15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"]

DATA_FOLDER_L =  'G:/3-ALL Python and AI/my codes/inputs/CSVBuy'    #   'C:/pythonFiles/inputs'
DATA_FOLDER ='G:/3-ALL Python and AI/my codes/input_Buy_file'   # 'C:/Users/Hossein/AppData/Roaming/MetaQuotes/Terminal/Common/Files/' 

INPUT_FILE = 'inputFile_Python2.csv'
INPUT_PATH = os.path.join(DATA_FOLDER, INPUT_FILE)

TRAIN_FILES = {tag: f"{tag}.csv" for tag in MODEL_TAGS}
TRAIN_PATHS = {tag: os.path.join(DATA_FOLDER_L, fname) for tag, fname in TRAIN_FILES.items()}

OUTPUT_FILES = {tag: f"prediction_{tag}.txt" for tag in MODEL_TAGS}
OUTPUT_PATHS = {tag: os.path.join(DATA_FOLDER, fname) for tag, fname in OUTPUT_FILES.items()}

EXCLUDE_COLUMNS = [
    'Supp_Z_TF 15 min=', 'Supp_Z_TF 1h=', 'Supp_Z_TF 4h=', 'Supp_Z_TF D=',
    'Ress_Z_TF 15 min=', 'Ress_Z_TF 1h=', 'Ress_Z_TF 4h=', 'Ress_Z_TF D=',
]

#    'n_bar_Change_ravand_TF 15 min=', 'n_bar_Change_ravand_TF 1h=', 'n_bar_Change_ravand_TF 4h=', 'n_bar_Change_ravand_TF D=',
#    'n_bar_Change_ravand_Zstr 15 min=', 'n_bar_Change_ravand_Zstr 1h=', 'n_bar_Change_ravand_Zstr 4h=', 'n_bar_Change_ravand_Zstr D=',
#    'now_ravand_perc_TF 15 min=', 'now_ravand_perc_TF 1h=', 'now_ravand_perc_TF 4h=', 'now_ravand_perc_TF D=',
#    'sec_ravand_perc_TF 15 min=', 'sec_ravand_perc_TF 1h=', 'sec_ravand_perc_TF 4h=', 'sec_ravand_perc_TF D=',
#    'third_ravand_perc_TF 15 min=', 'third_ravand_perc_TF 1h=', 'third_ravand_perc_TF 4h=', 'third_ravand_perc_TF D='

# ===== Data filtering function =====
def filter_data(df):
    # حذف نمونه‌هایی که ستون Sood_zarar مقدار ندارد
    df_filtered = df.dropna(subset=['Sood_zarar']).reset_index(drop=True)
    print(f"Samples after filtering: {len(df_filtered)}")
    return df_filtered

def analyze_profit_distribution(file_path, model_tag):
    data = pd.read_csv(file_path)
    data = filter_data(data)

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
        if col_name == 'Sood_zarar':
            return False
        if col_name in EXCLUDE_COLUMNS:
            return False
        return any(tf in col_name for tf in allowed_timeframes)

    filtered_cols = [col for col in data.columns if is_allowed_col(col)]
    filtered_data = data[filtered_cols]

    print(f"{model_tag} data shape: {filtered_data.shape}\n")

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
        data = filter_data(data)

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
            if col_name == 'Sood_zarar':
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
        self.y = data['Sood_zarar']
        return self.X, self.y

class ModelTrainer:
    def __init__(self, max_depth=5, random_state=42):
        self.model = XGBRegressor(max_depth=max_depth, random_state=random_state, eval_metric='rmse')

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)  # اصلاح این خط
        rmse = sqrt(mse)  # اصلاح این خط
        print(f'Max depth={self.model.get_params()["max_depth"]} Test RMSE: {rmse:.4f}')
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

    def predict(self, data_dict):
        X = self.preprocess_new_data(data_dict)
        pred_value = self.model.predict(X)[0]
        return pred_value

def read_input_file(filename):
    df = pd.read_csv(filename)
    return df.iloc[0].to_dict()

if __name__ == '__main__':
    print("Loading and training regression models...")

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
                    pred_value = predictor.predict(input_data)
                    output_path = OUTPUT_PATHS[tag]

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"{pred_value:.6f}")

                    print(f"{tag} prediction: {pred_value:.6f}")

                os.remove(INPUT_PATH)
                print("Prediction done. Waiting for next input...")

            except Exception as e:
                print("Error processing input:", e)

        time.sleep(1)
