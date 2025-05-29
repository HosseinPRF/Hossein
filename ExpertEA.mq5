#include <PredictionLoader.mqh>

//+------------------------------------------------------------------+
//| Expert inputs                                                    |
//+------------------------------------------------------------------+
input string data_folder = "G:\\3-ALL Python and AI\\my codes\\input_Buy_file\\";

string model_tags[] = {"15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"};

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // ساخت شیء کلاس و تنظیم پارامترها
   PredictionLoader loader;
   loader.SetDataFolder(data_folder);
   loader.SetModelTags(model_tags);

   // بارگذاری فایل‌ها در هر تیک
   bool loaded = loader.LoadAllPredictions();
   if(!loaded)
     {
      Print("❌ Error loading some predictions.");
      return; // اگر بارگذاری موفق نبود، ادامه نده
     }
   else
     {
      Print("✅ Predictions loaded successfully.");
     }

   // گرفتن احتمالات از کلاس
   double p_1h = loader.GetProfitProbByTag("1hBuy");
   double n_1h = loader.GetNeutralProbByTag("1hBuy");
   double l_1h = loader.GetLossProbByTag("1hBuy");

   double p_2h = loader.GetProfitProbByTag("2hBuy");
   double n_2h = loader.GetNeutralProbByTag("2hBuy");
   double l_2h = loader.GetLossProbByTag("2hBuy");

   double p_3h = loader.GetProfitProbByTag("3hBuy");
   double n_3h = loader.GetNeutralProbByTag("3hBuy");
   double l_3h = loader.GetLossProbByTag("3hBuy");

   double p_1d = loader.GetProfitProbByTag("1DBuy");
   double n_1d = loader.GetNeutralProbByTag("1DBuy");
   double l_1d = loader.GetLossProbByTag("1DBuy");

   // شرط خرید
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
