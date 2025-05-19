import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk, messagebox

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
        self.accuracy = None
        self.feature_importances_ = None
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.feature_importances_ = self.model.feature_importances_
        self.X_test = X_test
        self.y_test = y_test
        print(f'Accuracy on test set: {self.accuracy:.4f}')
    
    def get_model(self):
        return self.model

class AccuracyImportanceGUI:
    def __init__(self, accuracy, feature_importances, feature_names):
        self.root = tk.Tk()
        self.root.title("دقت مدل و اهمیت ویژگی‌ها")
        self.root.geometry("700x500")
        
        ttk.Label(self.root, text=f"دقت مدل روی داده تست: {accuracy:.4f}", font=("Arial", 16)).pack(pady=10)
        
        # مرتب‌سازی ویژگی‌ها بر اساس اهمیت
        fi_sorted = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
        
        # ساخت جدول نمایش اهمیت‌ها
        columns = ('Feature', 'Importance')
        tree = ttk.Treeview(self.root, columns=columns, show='headings')
        tree.heading('Feature', text='ویژگی')
        tree.heading('Importance', text='اهمیت')
        tree.column('Feature', width=400)
        tree.column('Importance', width=150, anchor='center')
        tree.pack(fill=tk.BOTH, expand=True)
        
        for feat, imp in fi_sorted:
            tree.insert('', tk.END, values=(feat, f"{imp:.5f}"))
        
        self.root.mainloop()

def read_input_file(filename):
    df = pd.read_csv(filename)
    return df.iloc[0].to_dict()

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

if __name__ == '__main__':
    train_file = 'buy.csv'
    input_file = 'input_buy.csv'
    
    loader = DataLoaderAndPreprocessor(train_file)
    X, y = loader.load_and_preprocess()
    
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    model = trainer.get_model()
    feature_cols = list(X.columns)
    
    predictor = Predictor(model, loader.label_encoders, loader.cat_columns, feature_cols)
    
    input_data = read_input_file(input_file)
    pred_class, pred_proba = predictor.predict(input_data)
    print(f"Prediction: {'سودده' if pred_class == 1 else 'ضررده'}, Probability: {pred_proba:.4f}")
    
    # باز کردن GUI برای نمایش دقت و اهمیت ویژگی‌ها
    gui = AccuracyImportanceGUI(trainer.accuracy, trainer.feature_importances_, feature_cols)
