#include <ZigZagTrendAnalyzer.mqh>

double ZigZag_Buffer_Zstr[];
double ZigZag_Buffer_F_Zstr[][2];
double ZigZag_Buffer_F2_Zstr[][2];

bool Strong_Signal_From_zigzag_Zstr[];
bool Signal_From_zigzag_Zstr[];
int n_bar_Change_ravand_Zstr[];
string Ravand_symbol_Zstr[];

int copied;  // تعداد داده‌ها، مثلاً کندل‌ها

void OnInit()
  {
   // مقداردهی اولیه سایز آرایه‌ها (مقدار copied را تنظیم کن)
   copied = 1000;  // مثال، سایز واقعی را تنظیم کن

   ArrayResize(ZigZag_Buffer_Zstr, copied);
   ArrayResize(ZigZag_Buffer_F_Zstr, 0);
   ArrayResize(ZigZag_Buffer_F2_Zstr, 0);
   ArrayResize(Strong_Signal_From_zigzag_Zstr, copied);
   ArrayResize(Signal_From_zigzag_Zstr, copied);
   ArrayResize(n_bar_Change_ravand_Zstr, copied);
   ArrayResize(Ravand_symbol_Zstr, copied);
  }

void OnTick()
  {
   // مثلا هر نماد یا کندل را پردازش کن
   for(int i=0; i<copied; i++)
     {
      double this_day_CP[]; // باید مقداردهی شود از دیتاهای قیمتی تو (Close Price)
      MqlRates mrate[];     // باید مقداردهی شود (دیتای کندل)

      // نمونه: مقداردهی این دیتاها طبق کد خودت انجام شود

      int symbol_size = copied; // یا سایز واقعی دیتا

      ZigZagTrendAnalyzer analyzer(ZigZag_Buffer_Zstr, ZigZag_Buffer_F_Zstr, ZigZag_Buffer_F2_Zstr,
                                  copied,
                                  Strong_Signal_From_zigzag_Zstr,
                                  Signal_From_zigzag_Zstr,
                                  n_bar_Change_ravand_Zstr,
                                  Ravand_symbol_Zstr);

      analyzer.SetLastPrice(0); // یا مقدار درست آخرین قیمت

      analyzer.Process(i, symbol_size, this_day_CP, mrate);

      analyzer.CalculateSupportResistance();

      int sup_count = analyzer.GetSupportCount();
      for(int idx=0; idx<sup_count; idx++)
        {
         double support_price = analyzer.GetSupportPrice(idx);
         double support_dist = analyzer.GetSupportDistance(idx);
         // اینجا می‌توانی حمایت‌ها را استفاده کنی
        }

      int res_count = analyzer.GetResistanceCount();
      for(int idx=0; idx<res_count; idx++)
        {
         double resistance_price = analyzer.GetResistancePrice(idx);
         double resistance_dist = analyzer.GetResistanceDistance(idx);
         // اینجا می‌توانی مقاومت‌ها را استفاده کنی
        }
     }
  }



        //-------------------------------استفاده دوبار از كلاس به جاي دوبار نوشتن همه چيز (مستقيم از چت جي پي تي كپي شده و درست و مرتب نيست )
        //-------------------------------        مثال ساده:
        //-------------------------------        فرض کن کلاس قبلی رو داری:


        ZigZagTrendAnalyzer analyzer1(ZigZag_Buffer_Zstr, ZigZag_Buffer_F_Zstr, ZigZag_Buffer_F2_Zstr,
                                    copied,
                                    Strong_Signal_From_zigzag_Zstr, Signal_From_zigzag_Zstr,
                                    n_bar_Change_ravand_Zstr, Ravand_symbol_Zstr);

        //-------------------------------        حالا تو فقط باید اینطوری جایگزین کنی:

        ZigZagTrendAnalyzer analyzer2(ZigZag_Buffer, ZigZag_Buffer_F, ZigZag_Buffer_F2,
                                    copied,
                                    Strong_Signal_From_zigzag, Signal_From_zigzag,
                                    n_bar_Change_ravand, Ravand_symbol);

        //-------------------------------        و بقیه کد بدون تغییر می‌مونه.