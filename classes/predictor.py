import pandas as pd

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
                    le_classes = list(le.classes_)
                    le_classes.append(val)
                    le.classes_ = le_classes
                    new_df[col] = le.transform(new_df[col])
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
