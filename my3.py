import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
        print(f'Accuracy on test set: {accuracy:.2f}')
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
                le = self.label_encoders[col]
                val = new_df.at[0, col]
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
        new_df = self.preprocess_new_data(data_dict)
        pred_class = self.model.predict(new_df)[0]
        pred_proba = self.model.predict_proba(new_df)[0][1]
        return pred_class, pred_proba

def read_single_line_input(file_path):
    df = pd.read_csv(file_path)
    # فرض کنیم فقط یک خط داده ورودی دارد
    return df.iloc[0].to_dict()

if __name__ == '__main__':
    # آموزش مدل با داده buy.csv
    file_train = 'buy.csv'
    loader = DataLoaderAndPreprocessor(file_train)
    X, y = loader.load_and_preprocess()
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    predictor = Predictor(trainer.get_model(), loader.label_encoders, loader.cat_columns, list(X.columns))
    
    # خواندن ورودی پیش‌بینی تک خطی
    file_input = 'input_buy.csv'  # فایل ورودی تک خطی که MQL5 می‌سازد
    input_data = read_single_line_input(file_input)
    
    # پیش‌بینی
    pred_class, pred_proba = predictor.predict(input_data)
    print(f'Prediction: {"سودده" if pred_class == 1 else "ضررده"}, Probability: {pred_proba:.3f}')
    
    # ذخیره احتمال در فایل خروجی
    with open('prediction_result.txt', 'w', encoding='utf-8') as f:
        f.write(f'{pred_proba:.6f}')
