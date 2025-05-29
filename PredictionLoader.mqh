//+------------------------------------------------------------------+
//| PredictionLoader.mqh                                             |
//+------------------------------------------------------------------+
class PredictionLoader
  {
private:
   string data_folder;
   string model_tags[];
   double profit_probs[];
   double neutral_probs[];
   double loss_probs[];

public:
   // سازنده
   void SetDataFolder(string folder) { data_folder = folder; }
   void SetModelTags(string &tags[]) { model_tags = tags; }
   
   // خواندن فایل پیش‌بینی
   bool ReadPredictionFile(string filename, double &profit_prob, double &neutral_prob, double &loss_prob)
     {
      int file_handle=FileOpen(filename, FILE_READ|FILE_TXT|FILE_ANSI);
      if(file_handle==INVALID_HANDLE)
        {
         Print("Cannot open file: ", filename);
         return(false);
        }
      string line=FileReadString(file_handle);
      FileClose(file_handle);

      string probs[];
      int count=StringSplit(line,' ',probs);
      if(count<3)
        {
         Print("File format error in ", filename);
         return(false);
        }

      profit_prob = StrToDouble(probs[0]);
      neutral_prob = StrToDouble(probs[1]);
      loss_prob = StrToDouble(probs[2]);

      return(true);
     }

   // بارگذاری همه پیش‌بینی‌ها
   bool LoadAllPredictions()
     {
      ArrayResize(profit_probs, ArraySize(model_tags));
      ArrayResize(neutral_probs, ArraySize(model_tags));
      ArrayResize(loss_probs, ArraySize(model_tags));

      bool all_ok = true;
      for(int i=0; i<ArraySize(model_tags); i++)
        {
         string file_path = data_folder + "prediction_" + model_tags[i] + ".txt";
         double p=0, n=0, l=0;
         if(ReadPredictionFile(file_path, p, n, l))
           {
            profit_probs[i] = p;
            neutral_probs[i] = n;
            loss_probs[i] = l;
            PrintFormat("Prediction %s: سودده=%.3f, بدون تغییر=%.3f, ضررده=%.3f", model_tags[i], p, n, l);
           }
         else
           {
            profit_probs[i] = neutral_probs[i] = loss_probs[i] = -1;
            all_ok = false;
           }
        }
      return all_ok;
     }

   // گرفتن احتمال سودده با تگ
   double GetProfitProbByTag(string tag)
     {
      for(int i=0; i<ArraySize(model_tags); i++)
        {
         if(model_tags[i] == tag)
            return profit_probs[i];
        }
      return -1;
     }

   // گرفتن احتمال بدون تغییر با تگ
   double GetNeutralProbByTag(string tag)
     {
      for(int i=0; i<ArraySize(model_tags); i++)
        {
         if(model_tags[i] == tag)
            return neutral_probs[i];
        }
      return -1;
     }

   // گرفتن احتمال ضررده با تگ
   double GetLossProbByTag(string tag)
     {
      for(int i=0; i<ArraySize(model_tags); i++)
        {
         if(model_tags[i] == tag)
            return loss_probs[i];
        }
      return -1;
     }
  };
