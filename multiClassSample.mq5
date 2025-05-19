//+------------------------------------------------------------------+
//| Expert Inputs                                                    |
//+------------------------------------------------------------------+
input string InpSymbol1 = "AMZN";           // نماد اول
input string InpSymbol2 = "AAPL";           // نماد دوم
input ENUM_TIMEFRAMES InpTF1 = PERIOD_H1;   // تایم فریم اول (1 ساعت)
input ENUM_TIMEFRAMES InpTF2 = PERIOD_D1;   // تایم فریم دوم (1 روز)
input int InpRSIPeriod = 14;                 // دوره RSI
input double InpRSIThreshold = 50.0;         // آستانه RSI برای مثبت/منفی
input int InpMACDFastEMA = 12;                // MACD Fast EMA
input int InpMACDSlowEMA = 26;                // MACD Slow EMA
input int InpMACDSignalSMA = 9;                // MACD Signal SMA
input double InpLots = 0.1;                    // حجم معامله ثابت
input int InpSlippage = 10;                     // اسلیپیج مجاز

//+------------------------------------------------------------------+
//| کلاس Logger برای ثبت گزارش داخل Journal                        |
//+------------------------------------------------------------------+
class Logger
{
public:
   static void Log(string message)
   {
      PrintFormat("[%s] %s", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), message);
   }
};

//+------------------------------------------------------------------+
//| کلاس IndicatorChecker برای بررسی سیگنال اندیکاتورها            |
//+------------------------------------------------------------------+
class IndicatorChecker
{
private:
   string symbol;
   ENUM_TIMEFRAMES timeframe;

   int rsi_period;
   double rsi_threshold;

   int macd_fast;
   int macd_slow;
   int macd_signal;

public:
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

   bool IsMACDPositive()
   {
      double macd_main[], macd_signal_buf[];
      int handle = iMACD(symbol, timeframe, macd_fast, macd_slow, macd_signal, PRICE_CLOSE);

      if(handle == INVALID_HANDLE)
      {
         Logger::Log("Failed to get MACD handle for " + symbol + " timeframe " + EnumToString(timeframe));
         return false;
      }

      if(CopyBuffer(handle, 0, 0, 1, macd_main) <= 0 ||
         CopyBuffer(handle, 1, 0, 1, macd_signal_buf) <= 0)
      {
         Logger::Log("Failed to copy MACD buffers for " + symbol + " timeframe " + EnumToString(timeframe));
         return false;
      }

      IndicatorRelease(handle);

      return (macd_main[0] > macd_signal_buf[0]);
   }

   bool IsRSINegative() { return !IsRSIPositive(); }
   bool IsMACDNegative() { return !IsMACDPositive(); }
};

//+------------------------------------------------------------------+
//| کلاس SymbolAnalyzer برای تحلیل روند نماد در چند تایم فریم       |
//+------------------------------------------------------------------+
class SymbolAnalyzer
{
private:
   string symbol;
   ENUM_TIMEFRAMES tf1;
   ENUM_TIMEFRAMES tf2;

   IndicatorChecker *ic_tf1;
   IndicatorChecker *ic_tf2;

public:
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

   ~SymbolAnalyzer()
   {
      delete ic_tf1;
      delete ic_tf2;
   }

   bool IsTrendPositive()
   {
      return (ic_tf1.IsRSIPositive() && ic_tf2.IsRSIPositive() &&
              ic_tf1.IsMACDPositive() && ic_tf2.IsMACDPositive());
   }

   bool IsTrendNegative()
   {
      return (ic_tf1.IsRSINegative() && ic_tf2.IsRSINegative() &&
              ic_tf1.IsMACDNegative() && ic_tf2.IsMACDNegative());
   }
};

//+------------------------------------------------------------------+
//| کلاس TradeManager برای مدیریت پوزیشن ها                         |
//+------------------------------------------------------------------+
class TradeManager
{
private:
   double lots;
   int slippage;

public:
   TradeManager(double lot_size, int slip)
   {
      lots = lot_size;
      slippage = slip;
   }

   bool ClosePosition(ulong ticket)
   {
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

      if(type == POSITION_TYPE_BUY)
         request.type = ORDER_TYPE_SELL;
      else if(type == POSITION_TYPE_SELL)
         request.type = ORDER_TYPE_BUY;
      else
      {
         Logger::Log("ClosePosition: Unknown position type for ticket=" + IntegerToString(ticket));
         return false;
      }

      request.price = (request.type == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);

      if(!OrderSend(request, result))
      {
         Logger::Log("ClosePosition: OrderSend failed for ticket=" + IntegerToString(ticket) + " Error=" + IntegerToString(GetLastError()));
         return false;
      }

      if(result.retcode != TRADE_RETCODE_DONE)
      {
         Logger::Log("ClosePosition: OrderSend retcode not done for ticket=" + IntegerToString(ticket) + " RetCode=" + IntegerToString(result.retcode));
         return false;
      }

      Logger::Log("Position closed successfully, ticket=" + IntegerToString(ticket));
      return true;
   }

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

      if(!OrderSend(request, result))
      {
         Logger::Log("OpenPosition: OrderSend failed for symbol=" + symbol + " Error=" + IntegerToString(GetLastError()));
         return false;
      }

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
//| کلاس Expert اصلی                                                 |
//+------------------------------------------------------------------+
class Expert
{
private:
   SymbolAnalyzer *analyzer1;
   SymbolAnalyzer *analyzer2;
   TradeManager *tradeManager;

   string symbol1;
   string symbol2;

public:
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

   ~Expert()
   {
      delete analyzer1;
      delete analyzer2;
      delete tradeManager;
   }

   void OnTick()
   {
      ManageSymbol(symbol1, analyzer1);
      ManageSymbol(symbol2, analyzer2);
   }

private:
   void ManageSymbol(string symbol, SymbolAnalyzer *analyzer)
   {
      int total_positions = PositionsTotal();
      bool has_long = false;
      bool has_short = false;
      ulong long_ticket = 0;
      ulong short_ticket = 0;

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

      bool trend_pos = analyzer.IsTrendPositive();
      bool trend_neg = analyzer.IsTrendNegative();

      // منطق باز و بسته کردن پوزیشن‌ها
      if(trend_pos && !has_long)
      {
         if(has_short)
            tradeManager.ClosePosition(short_ticket);

         tradeManager.OpenPosition(symbol, ORDER_TYPE_BUY);
      }
      else if(trend_neg && !has_short)
      {
         if(has_long)
            tradeManager.ClosePosition(long_ticket);

         tradeManager.OpenPosition(symbol, ORDER_TYPE_SELL);
      }
      else if(has_long && !trend_pos)
      {
         tradeManager.ClosePosition(long_ticket);
      }
      else if(has_short && !trend_neg)
      {
         tradeManager.ClosePosition(short_ticket);
      }
   }
};

//+------------------------------------------------------------------+
//| Global variable                                                  |
//+------------------------------------------------------------------+
Expert *expert = NULL;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
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
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   delete expert;
   expert = NULL;
   Logger::Log("Expert Stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(expert != NULL)
      expert.OnTick();
}
