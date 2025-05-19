from data_loader import DataLoaderAndPreprocessor
from model_trainer import ModelTrainer
from predictor import Predictor

if __name__ == '__main__':
    file_path = 'buy.csv'
    
    # بارگذاری و پیش‌پردازش داده‌ها
    loader = DataLoaderAndPreprocessor(file_path)
    X, y = loader.load_and_preprocess()
    
    # آموزش مدل
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    # ساخت پیش‌بینی‌کننده
    predictor = Predictor(trainer.get_model(), loader.label_encoders, loader.cat_columns, list(X.columns))
    
    # نمونه داده جدید (حتما همه ستون‌ها رو اضافه کن یا با مقادیر مناسب جایگزین کن)
    sample_input = {
        'Ravand_TF 15 min=': 'Ravand mobham_Shayad_Kanal',
        'Ravand_TF 1h=': 'Ravand_Transient Soodi To Nozooli',
        'Ravand_TF 4h=': 'Ravand mobham_Shayad_Kanal',
        'Ravand_TF D=': 'Ravand_Transient Soodi To Nozooli',
        'RSI 15 min=': 50,
        'MACD_Status 15 min=': 0.3,
        'third_ravand_perc_TF 1h=': -1.5,
        'n_bar_Change_ravand_TF 15 min=': 2,
        'PTL_Status 15 min=': 1,
        'Ravand_Zstr_TF 15 min=': 2,
        'third_ravand_perc_TF 4h=': 3,
        'Supp_Z_TF 1h=': 1,
        'n_bar_Change_ravand_Zstr 15 min=': 5,
        'now_ravand_perc_TF 4h=': 1.5,
        'n_bar_Change_ravand_TF 4h=': 4,
        'RSI 4h=': 40,
        # ستون های دیگر را هم اضافه کن یا مقدار بده
    }
    
    pred_class, pred_proba = predictor.predict(sample_input)
    print(f'Prediction: {"سودده" if pred_class == 1 else "ضررده"}, Probability of profit: {pred_proba:.2f}')
