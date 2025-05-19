//+------------------------------------------------------------------+
//| Expert Inputs                                                    |
//| ورودی‌های اکسپرت: نمادها، تایم فریم‌ها، پارامترهای اندیکاتورها،|
//| حجم معاملات و اسلیپیج                                           |
//+------------------------------------------------------------------+
input string InpSymbol1 = "AMZN";           // نماد اول (مثلاً Amazon)
input string InpSymbol2 = "AAPL";           // نماد دوم (مثلاً Apple)
input ENUM_TIMEFRAMES InpTF1 = PERIOD_H1;   // تایم فریم اول (یک ساعت)
input ENUM_TIMEFRAMES InpTF2 = PERIOD_D1;   // تایم فریم دوم (یک روز)
input int InpRSIPeriod = 14;                 // دوره زمانی اندیکاتور RSI
input double InpRSIThreshold = 50.0;         // آستانه RSI برای سیگنال مثبت یا منفی
input int InpMACDFastEMA = 12;                // پارامتر EMA سریع برای MACD
input int InpMACDSlowEMA = 26;                // پارامتر EMA کند برای MACD
input int InpMACDSignalSMA = 9;                // پارامتر SMA سیگنال MACD
input double InpLots = 0.1;                    // حجم معامله ثابت
input int InpSlippage = 10;                     // حداکثر اسلیپیج مجاز برای سفارش‌ها

//+------------------------------------------------------------------+
//| کلاس Logger برای ثبت پیام‌ها و گزارش‌ها در Journal متاتریدر    |
//| این کلاس یک متد استاتیک برای چاپ پیام همراه با زمان دقیق دارد  |
//+------------------------------------------------------------------+
class Logger
{
public:
   // ثبت پیام در لاگ همراه با زمان فعلی
   static void Log(string message)
   {
      PrintFormat("[%s] %s", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), message);
   }
};

//+------------------------------------------------------------------+
//| کلاس IndicatorChecker برای بررسی سیگنال‌های اندیکاتور RSI و MACD|
//| این کلاس وظیفه خواندن مقادیر اندیکاتورها در تایم فریم مشخص دارد|
//+------------------------------------------------------------------+
class IndicatorChecker
{
private:
   string symbol;                 // نماد مورد بررسی
   ENUM_TIMEFRAMES timeframe;    // تایم فریم مورد بررسی

   int rsi_period;               // دوره RSI
   double rsi_threshold;         // آستانه RSI برای تعیین مثبت یا منفی

   int macd_fast;                // پارامتر EMA سریع MACD
   int macd_slow;                // پارامتر EMA کند MACD
   int macd_signal;              // پارامتر SMA سیگنال MACD

public:
   // سازنده کلاس با تنظیم پارامترهای ورودی
   IndicatorChecker(string sym, ENUM_TIMEFRAMES tf,
                    int rsi_p, double rsi_th,
                    int macd_f, int macd_s, int macd_sig)
   {
      symbol = sym;
      timeframe = tf;
      rsi_period = rsi_p;
      rsi_threshold = rsi_th;
      macd_fast = macd_f;
      macd_slow = macd_s;
      macd_signal = macd_sig;
   }

   // تابع بررسی سیگنال مثبت RSI (بالاتر از آستانه)
   bool IsRSIPositive()
   {
      double rsi = iRSI(symbol, timeframe, rsi_period, PRICE_CLOSE, 0);
      if(rsi == WRONG_VALUE)
      {
         Logger::Log("Failed to get RSI for " + symbol + " timeframe " + EnumToString(timeframe));
         return false;
      }
      return (rsi > rsi_threshold);
   }

   // تابع بررسی سیگنال مثبت MACD (MACD بالاتر از خط سیگنال)
   bool IsMACDPositive()
   {
      double macd_main[], macd_signal_buf[];
      int handle = iMACD(symbol, timeframe, macd_fast, macd_slow, macd_signal, PRICE_CLOSE);

      if(handle == INVALID_HANDLE)
      {
         Logger::Log("Failed to get MACD handle for " + symbol + " timeframe " + EnumToString(timeframe));
         return false;
      }

      // گرفتن مقدار MACD اصلی و سیگنال از بافرهای اندیکاتور
      if(CopyBuffer(handle, 0, 0, 1, macd_main) <= 0 ||
         CopyBuffer(handle, 1, 0, 1, macd_signal_buf) <= 0)
      {
         Logger::Log("Failed to copy MACD buffers for " + symbol + " timeframe " + EnumToString(timeframe));
         IndicatorRelease(handle);
         return false;
      }

      IndicatorRelease(handle);

      // سیگنال مثبت وقتی است که مقدار MACD بالاتر از سیگنال باشد
      return (macd_main[0] > macd_signal_buf[0]);
   }

   // تابع بررسی سیگنال منفی RSI (برعکس مثبت)
   bool IsRSINegative() { return !IsRSIPositive(); }

   // تابع بررسی سیگنال منفی MACD (برعکس مثبت)
   bool IsMACDNegative() { return !IsMACDPositive(); }
};

//+------------------------------------------------------------------+
//| کلاس SymbolAnalyzer برای تحلیل روند کلی یک نماد                  |
//| این کلاس با استفاده از دو نمونه IndicatorChecker برای دو تایم   |
//| فریم مختلف و با استفاده از سیگنال‌های RSI و MACD، روند کلی را   |
//| تعیین می‌کند.                                                    |
//+------------------------------------------------------------------+
class SymbolAnalyzer
{
private:
   string symbol;               // نماد مورد بررسی
   ENUM_TIMEFRAMES tf1;         // تایم فریم اول
   ENUM_TIMEFRAMES tf2;         // تایم فریم دوم

   IndicatorChecker *ic_tf1;    // اندیکاتور برای تایم فریم اول
   IndicatorChecker *ic_tf2;    // اندیکاتور برای تایم فریم دوم

public:
   // سازنده کلاس با مقداردهی نمونه‌های IndicatorChecker
   SymbolAnalyzer(string sym, ENUM_TIMEFRAMES t1, ENUM_TIMEFRAMES t2,
                  int rsi_p, double rsi_th,
                  int macd_f, int macd_s, int macd_sig)
   {
      symbol = sym;
      tf1 = t1;
      tf2 = t2;
      ic_tf1 = new IndicatorChecker(symbol, tf1, rsi_p, rsi_th, macd_f, macd_s, macd_sig);
      ic_tf2 = new IndicatorChecker(symbol, tf2, rsi_p, rsi_th, macd_f, macd_s, macd_sig);
   }

   // مخرب کلاس برای آزادسازی حافظه
   ~SymbolAnalyzer()
   {
      delete ic_tf1;
      delete ic_tf2;
   }

   // بررسی روند مثبت: هر دو اندیکاتور RSI و MACD در هر دو تایم فریم مثبت باشند
   bool IsTrendPositive()
   {
      return (ic_tf1.IsRSIPositive() && ic_tf2.IsRSIPositive() &&
              ic_tf1.IsMACDPositive() && ic_tf2.IsMACDPositive());
   }

   // بررسی روند منفی: هر دو اندیکاتور RSI و MACD در هر دو تایم فریم منفی باشند
   bool IsTrendNegative()
   {
      return (ic_tf1.IsRSINegative() && ic_tf2.IsRSINegative() &&
              ic_tf1.IsMACDNegative() && ic_tf2.IsMACDNegative());
   }
};

//+------------------------------------------------------------------+
//| کلاس TradeManager برای مدیریت باز و بسته کردن معاملات           |
//| شامل بازکردن پوزیشن لانگ و شورت و بستن موقعیت‌ها با کنترل خطا  |
//+------------------------------------------------------------------+
class TradeManager
{
private:
   double lots;      // حجم معامله
   int slippage;     // اسلیپیج مجاز

public:
   // سازنده با تنظیم حجم و اسلیپیج
   TradeManager(double lot_size, int slip)
   {
      lots = lot_size;
      slippage = slip;
   }

   // بستن پوزیشن بر اساس تیکت (شناسه)
   bool ClosePosition(ulong ticket)
   {
      // انتخاب پوزیشن
      if(!PositionSelectByTicket(ticket))
      {
         Logger::Log("ClosePosition: Position not found, ticket=" + IntegerToString(ticket));
         return false;
      }

      string symbol = PositionGetString(POSITION_SYMBOL);
      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      MqlTradeRequest request={0};
      MqlTradeResult result={0};

      request.action = TRADE_ACTION_DEAL;
      request.position = ticket;
      request.symbol = symbol;
      request.volume = volume;
      request.deviation = slippage;

      // اگر پوزیشن لانگ است، باید سفارش فروش ارسال شود و برعکس
      if(type == POSITION_TYPE_BUY)
         request.type = ORDER_TYPE_SELL;
      else if(type == POSITION_TYPE_SELL)
         request.type = ORDER_TYPE_BUY;
      else
      {
         Logger::Log("ClosePosition: Unknown position type for ticket=" + IntegerToString(ticket));
         return false;
      }

      // تعیین قیمت مناسب برای سفارش بستن
      request.price = (request.type == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);

      // ارسال درخواست معامله
      if(!OrderSend(request, result))
      {
         Logger::Log("ClosePosition: OrderSend failed for ticket=" + IntegerToString(ticket) + " Error=" + IntegerToString(GetLastError()));
         return false;
      }

      // بررسی نتیجه اجرای سفارش
      if(result.retcode != TRADE_RETCODE_DONE)
      {
         Logger::Log("ClosePosition: OrderSend retcode not done for ticket=" + IntegerToString(ticket) + " RetCode=" + IntegerToString(result.retcode));
         return false;
      }

      Logger::Log("Position closed successfully, ticket=" + IntegerToString(ticket));
      return true;
   }

   // باز کردن پوزیشن جدید با نوع مشخص (لانگ یا شورت)
   bool OpenPosition(string symbol, ENUM_ORDER_TYPE type)
   {
      MqlTradeRequest request={0};
      MqlTradeResult result={0};

      request.action = TRADE_ACTION_DEAL;
      request.symbol = symbol;
      request.volume = lots;
      request.type = type;
      request.price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
      request.deviation = slippage;

      // ارسال درخواست معامله
      if(!OrderSend(request, result))
      {
         Logger::Log("OpenPosition: OrderSend failed for symbol=" + symbol + " Error=" + IntegerToString(GetLastError()));
         return false;
      }

      // بررسی نتیجه اجرای سفارش
      if(result.retcode != TRADE_RETCODE_DONE)
      {
         Logger::Log("OpenPosition: OrderSend retcode not done for symbol=" + symbol + " RetCode=" + IntegerToString(result.retcode));
         return false;
      }

      Logger::Log("Position opened successfully: Symbol=" + symbol + " Type=" + EnumToString(type));
      return true;
   }
};

//+------------------------------------------------------------------+
//| کلاس اصلی Expert که نقطه شروع اکسپرت است و همه چیز را کنترل می‌کند|
//| شامل ساخت نمونه‌های SymbolAnalyzer و TradeManager و منطق کلی  |
//+------------------------------------------------------------------+
class Expert
{
private:
   SymbolAnalyzer *analyzer1;     // تحلیل‌گر نماد اول
   SymbolAnalyzer *analyzer2;     // تحلیل‌گر نماد دوم
   TradeManager *tradeManager;    // مدیر معاملات

   string symbol1;                // نام نماد اول
   string symbol2;                // نام نماد دوم

public:
   // سازنده با مقداردهی اولیه تمام کلاس‌ها و پارامترها
   Expert(string sym1, string sym2, double lots, int slippage,
          int rsi_p, double rsi_th,
          int macd_f, int macd_s, int macd_sig)
   {
      symbol1 = sym1;
      symbol2 = sym2;

      analyzer1 = new SymbolAnalyzer(symbol1, InpTF1, InpTF2, rsi_p, rsi_th, macd_f, macd_s, macd_sig);
      analyzer2 = new SymbolAnalyzer(symbol2, InpTF1, InpTF2, rsi_p, rsi_th, macd_f, macd_s, macd_sig);

      tradeManager = new TradeManager(lots, slippage);
   }

   // مخرب برای آزادسازی حافظه اختصاص داده شده
   ~Expert()
   {
      delete analyzer1;
      delete analyzer2;
      delete tradeManager;
   }

   // تابع اصلی که در هر تیک اجرا می‌شود
   void OnTick()
   {
      ManageSymbol(symbol1, analyzer1);
      ManageSymbol(symbol2, analyzer2);
   }

private:
   // مدیریت وضعیت و پوزیشن‌های هر نماد بر اساس تحلیل روند
   void ManageSymbol(string symbol, SymbolAnalyzer *analyzer)
   {
      int total_positions = PositionsTotal();
      bool has_long = false;
      bool has_short = false;
      ulong long_ticket = 0;
      ulong short_ticket = 0;

      // جستجو و بررسی پوزیشن‌های باز روی نماد مورد نظر
      for(int i=0; i<total_positions; i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionGetString(POSITION_SYMBOL) == symbol)
         {
            ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            if(type == POSITION_TYPE_BUY)
            {
               has_long = true;
               long_ticket = ticket;
            }
            else if(type == POSITION_TYPE_SELL)
            {
               has_short = true;
               short_ticket = ticket;
            }
         }
      }

      // دریافت سیگنال‌های روند از تحلیل‌گر
      bool trend_pos = analyzer.IsTrendPositive();
      bool trend_neg = analyzer.IsTrendNegative();

      // اگر روند مثبت است و پوزیشن لانگ نداریم، لانگ باز کن
      if(trend_pos && !has_long)
      {
         // اگر پوزیشن شورت باز است، ابتدا آن را ببند
         if(has_short)
            tradeManager.ClosePosition(short_ticket);

         // باز کردن پوزیشن لانگ
         tradeManager.OpenPosition(symbol, ORDER_TYPE_BUY);
      }
      // اگر روند منفی است و پوزیشن شورت نداریم، شورت باز کن
      else if(trend_neg && !has_short)
      {
         // اگر پوزیشن لانگ باز است، ابتدا آن را ببند
         if(has_long)
            tradeManager.ClosePosition(long_ticket);

         // باز کردن پوزیشن شورت
         tradeManager.OpenPosition(symbol, ORDER_TYPE_SELL);
      }
      // اگر پوزیشن لانگ داریم ولی روند مثبت نیست، پوزیشن را ببند
      else if(has_long && !trend_pos)
      {
         tradeManager.ClosePosition(long_ticket);
      }
      // اگر پوزیشن شورت داریم ولی روند منفی نیست، پوزیشن را ببند
      else if(has_short && !trend_neg)
      {
         tradeManager.ClosePosition(short_ticket);
      }
   }
};

//+------------------------------------------------------------------+
//| متغیر سراسری اکسپرت                                            |
//+------------------------------------------------------------------+
Expert *expert = NULL;

//+------------------------------------------------------------------+
//| تابع شروع اکسپرت                                                |
//| ایجاد نمونه Expert و مقداردهی اولیه                            |
//+------------------------------------------------------------------+
int OnInit()
{
   Logger::Log("Expert Started");
   expert = new Expert(InpSymbol1, InpSymbol2, InpLots, InpSlippage,
                       InpRSIPeriod, InpRSIThreshold,
                       InpMACDFastEMA, InpMACDSlowEMA, InpMACDSignalSMA);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| تابع توقف اکسپرت                                               |
//| آزادسازی حافظه و ثبت لاگ                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   delete expert;
   expert = NULL;
   Logger::Log("Expert Stopped");
}

//+------------------------------------------------------------------+
//| تابع تیک اکسپرت                                               |
//| در هر تیک، تابع OnTick کلاس Expert فراخوانی می‌شود             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(expert != NULL)
      expert.OnTick();
}
