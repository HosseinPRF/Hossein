import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
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
        self.rules = None
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy on test set: {accuracy:.2f}')
        self.rules = export_text(self.model, feature_names=list(X.columns))
        print('Decision Tree Rules:')
        print(self.rules)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
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

class CombinedGUI:
    def __init__(self, predictor, X_train, cat_columns, label_encoders, feature_columns):
        self.predictor = predictor
        self.X_train = X_train.reset_index(drop=True)
        self.cat_columns = cat_columns
        self.label_encoders = label_encoders
        self.feature_columns = feature_columns
        
        self.root = tk.Tk()
        self.root.title("پیش‌بینی سوددهی و نمایش نمونه‌ها")
        self.root.geometry("1000x650")
        
        # Canvas و Scrollbar برای فرم ورودی
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.input_frame = ttk.Frame(self.canvas)
        self.canvas_frame = self.canvas.create_window((0,0), window=self.input_frame, anchor='nw')
        self.input_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        ttk.Label(self.input_frame, text="ورودی‌های مدل را ویرایش کنید:").grid(row=0, column=0, columnspan=2, pady=5)
        
        self.entries = {}
        for i, col in enumerate(self.feature_columns):
            ttk.Label(self.input_frame, text=col).grid(row=i+1, column=0, sticky='w', padx=5, pady=2)
            entry = ttk.Entry(self.input_frame, width=40)
            entry.grid(row=i+1, column=1, pady=2, padx=5)
            self.entries[col] = entry
        
        # مقداردهی پیش‌فرض ورودی‌ها
        for col in self.entries:
            if col in self.cat_columns:
                self.entries[col].insert(0, self.label_encoders[col].classes_[0])
            else:
                self.entries[col].insert(0, "0")
        
        # دکمه پیش‌بینی
        pred_btn = ttk.Button(self.root, text="پیش‌بینی کن", command=self.run_prediction)
        pred_btn.pack(pady=5)
        
        # برچسب نتیجه پیش‌بینی
        self.result_label = ttk.Label(self.root, text="", font=('Arial', 14))
        self.result_label.pack(pady=5)
        
        # دکمه نمایش ۱۰ نمونه برتر
        show_top_btn = ttk.Button(self.root, text="نمایش ۱۰ نمونه با بیشترین احتمال سوددهی", command=self.show_top10)
        show_top_btn.pack(pady=10)
        
        # فریم جدول نمونه‌ها
        self.table_frame = ttk.Frame(self.root)
        self.table_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = None
        self.v_scroll = None
        self.h_scroll = None
    
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_frame, width=canvas_width)

    def run_prediction(self):
        input_data = {}
        for col, entry in self.entries.items():
            val = entry.get()
            if val == '':
                input_data[col] = 0
            else:
                try:
                    input_data[col] = float(val)
                except ValueError:
                    input_data[col] = val
        try:
            pred_class, pred_proba = self.predictor.predict(input_data)
            text = f'پیش‌بینی: {"سودده" if pred_class == 1 else "ضررده"}\nاحتمال سوددهی: {pred_proba:.2f}'
            self.result_label.config(text=text, foreground='green' if pred_class==1 else 'red')
        except Exception as e:
            messagebox.showerror("خطا در پیش‌بینی", str(e))

    def show_top10(self):
        probs = self.predictor.model.predict_proba(self.X_train)[:, 1]
        preds = self.predictor.model.predict(self.X_train)
        df = self.X_train.copy()
        df['Probability'] = probs
        df['Prediction'] = preds
        top10 = df.sort_values('Probability', ascending=False).head(10)

        # حذف ویجت‌های قبلی
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        columns = list(top10.columns)
        
        # فریم نگهدارنده جدول و اسکرول بار عمودی
        table_container = ttk.Frame(self.table_frame)
        table_container.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(table_container, columns=columns, show='headings')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = ttk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.h_scroll.pack(fill=tk.X)

        self.tree.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, anchor=tk.CENTER)

        for _, row in top10.iterrows():
            values = []
            for col in columns:
                v = row[col]
                if isinstance(v, float):
                    values.append(f"{v:.3f}")
                else:
                    values.append(str(v))
            self.tree.insert('', tk.END, values=values)

        # رویداد دوبار کلیک روی ردیف
        self.tree.bind("<Double-1>", self.on_double_click)

    def on_double_click(self, event):
        item_id = self.tree.focus()
        if not item_id:
            return
        values = self.tree.item(item_id)['values']
        columns = self.tree['columns']
        
        for col, val in zip(columns, values):
            if col in self.entries:
                if col in self.cat_columns:
                    le = self.label_encoders[col]
                    try:
                        idx = int(float(val)) if isinstance(val, str) else int(val)
                        if 0 <= idx < len(le.classes_):
                            val = le.classes_[idx]
                        else:
                            val = ''
                    except:
                        val = ''
                self.entries[col].delete(0, tk.END)
                self.entries[col].insert(0, val)

        self.result_label.config(text="مقادیر نمونه انتخاب شد. اکنون می‌توانید ویرایش و پیش‌بینی کنید.", foreground='blue')

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    file_path = 'buy.csv'  # مسیر فایل داده خود را تنظیم کنید
    loader = DataLoaderAndPreprocessor(file_path)
    X, y = loader.load_and_preprocess()
    
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    predictor = Predictor(trainer.get_model(), loader.label_encoders, loader.cat_columns, list(X.columns))
    
    gui = CombinedGUI(predictor, trainer.X_train, loader.cat_columns, loader.label_encoders, list(X.columns))
    gui.run()
