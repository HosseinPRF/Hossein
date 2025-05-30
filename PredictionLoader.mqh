//+------------------------------------------------------------------+
//| PredictionLoader.mqh                                             |
//+------------------------------------------------------------------+
class PredictionLoader
  {
private:
   string   model_tags[];
   double   profit_probs[];
   double   neutral_probs[];
   double   loss_probs[];

public:
   // تنظیم لیست تگ‌ها
   void SetModelTags(string &tags[])
     {
      ArrayResize(model_tags, ArraySize(tags));
      for(int i = 0; i < ArraySize(tags); i++)
         model_tags[i] = tags[i];
     }

   // خواندن فایل پیش‌بینی (از پوشه FILE_COMMON)
   bool ReadPredictionFile(string filename, double &profit_prob, double &neutral_prob, double &loss_prob)
     {
      int file_handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if(file_handle == INVALID_HANDLE)
        {
         int err = GetLastError();
         PrintFormat("❌ Cannot open file: %s | Error code: %d", filename, err);
         return false;
        }

      string line = FileReadString(file_handle);
      FileClose(file_handle);

      string probs[];
      int count = StringSplit(line, ' ', probs);
      if(count < 3)
        {
         Print("❌ File format error in ", filename);
         return false;
        }

      profit_prob  = StringToDouble(probs[0]);
      neutral_prob = StringToDouble(probs[1]);
      loss_prob    = StringToDouble(probs[2]);

      return true;
     }

   // بارگذاری تمام فایل‌ها
   bool LoadAllPredictions()
     {
      int size = ArraySize(model_tags);
      ArrayResize(profit_probs, size);
      ArrayResize(neutral_probs, size);
      ArrayResize(loss_probs, size);

      bool all_ok = true;
      for(int i = 0; i < size; i++)
        {
         string file_path = "prediction_" + model_tags[i] + ".txt";
         double p = 0, n = 0, l = 0;
         if(ReadPredictionFile(file_path, p, n, l))
           {
            profit_probs[i]  = p;
            neutral_probs[i] = n;
            loss_probs[i]    = l;
            PrintFormat("📊 Prediction %s: سودده=%.3f, بدون تغییر=%.3f, ضررده=%.3f", model_tags[i], p, n, l);
           }
         else
           {
            profit_probs[i]  = -1;
            neutral_probs[i] = -1;
            loss_probs[i]    = -1;
            all_ok = false;
           }
        }
      return all_ok;
     }

   // گرفتن احتمال با تگ
   double GetProfitProbByTag(string tag)
     {
      for(int i = 0; i < ArraySize(model_tags); i++)
         if(model_tags[i] == tag)
            return profit_probs[i];
      return -1;
     }

   double GetNeutralProbByTag(string tag)
     {
      for(int i = 0; i < ArraySize(model_tags); i++)
         if(model_tags[i] == tag)
            return neutral_probs[i];
      return -1;
     }

   double GetLossProbByTag(string tag)
     {
      for(int i = 0; i < ArraySize(model_tags); i++)
         if(model_tags[i] == tag)
            return loss_probs[i];
      return -1;
     }
  };
