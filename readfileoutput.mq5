//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
input string data_folder = "G:\\3-ALL Python and AI\\my codes\\input_Buy_file\\";

string model_tags[] = {"15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"};

// آرایه‌های ذخیره احتمالات برای هر تایم فریم و هر کلاس
double profit_probs[];   // احتمال سودده
double neutral_probs[];  // احتمال بدون تغییر
double loss_probs[];     // احتمال ضررده

// تابع خواندن فایل و گرفتن سه احتمال سودده، بدون تغییر، ضررده
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

   profit_prob = StrToDouble(probs[0]);  // سودده
   neutral_prob = StrToDouble(probs[1]); // بدون تغییر
   loss_prob = StrToDouble(probs[2]);    // ضررده

   return(true);
  }

// تابع خواندن همه فایل‌ها و ذخیره احتمالات در آرایه‌ها
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
         profit_probs[i] = neutral_probs[i] = loss_probs[i] = -1; // مقدار نامعتبر
         all_ok = false;
        }
     }
   return all_ok;
  }

// تابع برای گرفتن احتمال سودده تایم فریم خاص
double GetProfitProbByTag(string tag)
  {
   for(int i=0; i<ArraySize(model_tags); i++)
     {
      if(model_tags[i] == tag)
         return profit_probs[i];
     }
   return -1;
  }

// تابع برای گرفتن احتمال بدون تغییر تایم فریم خاص
double GetNeutralProbByTag(string tag)
  {
   for(int i=0; i<ArraySize(model_tags); i++)
     {
      if(model_tags[i] == tag)
         return neutral_probs[i];
     }
   return -1;
  }

// تابع برای گرفتن احتمال ضررده تایم فریم خاص
double GetLossProbByTag(string tag)
  {
   for(int i=0; i<ArraySize(model_tags); i++)
     {
      if(model_tags[i] == tag)
         return loss_probs[i];
     }
   return -1;
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   static bool data_loaded = false;

   if(!data_loaded)
     {
      if(LoadAllPredictions())
        {
         data_loaded = true;
         Print("✅ All predictions loaded successfully.");
        }
      else
        {
         Print("❌ Error loading some predictions.");
        }
     }

   // حالا می‌تونی هر سه احتمال هر تایم فریم رو داشته باشی و شرط دلخواهت رو اینجا بنویسی

   double p_1h = GetProfitProbByTag("1hBuy");
   double n_1h = GetNeutralProbByTag("1hBuy");
   double l_1h = GetLossProbByTag("1hBuy");

   double p_2h = GetProfitProbByTag("2hBuy");
   double n_2h = GetNeutralProbByTag("2hBuy");
   double l_2h = GetLossProbByTag("2hBuy");

   double p_3h = GetProfitProbByTag("3hBuy");
   double n_3h = GetNeutralProbByTag("3hBuy");
   double l_3h = GetLossProbByTag("3hBuy");

   double p_1d = GetProfitProbByTag("1DBuy");
   double n_1d = GetNeutralProbByTag("1DBuy");
   double l_1d = GetLossProbByTag("1DBuy");

   // مثال شرط خرید: 1h, 2h, 3h سودده بالا، 1d ضررده بالا
   if(p_1h > 0.5 && p_2h > 0.5 && p_3h > 0.5 && l_1d > 0.5)
     {
      Print("💡 Buy condition met: 1h,2h,3h سودده بالا و 1d ضررده بالا");

      if(PositionSelect(_Symbol) == false)
        {
         double lot = 0.1;
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         int ticket = OrderSend(_Symbol, ORDER_TYPE_BUY, lot, price, 3, 0, 0, "Buy signal from ML", 0, 0, clrGreen);
         if(ticket > 0)
            Print("Buy order placed, ticket: ", ticket);
         else
            Print("Failed to place buy order, error: ", GetLastError());
        }
      else
        {
         Print("Already have position, no new order.");
        }
     }
  }
