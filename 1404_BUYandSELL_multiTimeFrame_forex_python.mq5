//+------------------------------------------------------------------+
//|                                                   RSI_OBV_MACD.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

///HOSSEIN: adress zakhire file ha : C:\Users\Hossein\AppData\Roaming\MetaQuotes\Terminal\Common\Files


// on tester C:\Users\Hossein\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\MQL5\Files
//           C:\Users\Hossein\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files
//      python output files     C:\Users\Hossein\AppData\Roaming\MetaQuotes\Terminal\Common\Files\



#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "10.00"
#include <MarketBook.mqh>     // Include CMarketBook class
#property tester_file "FULL_3_99.csv"
#property tester_file "My_Portfolio.csv"
//#include <WinAPI\WinUser.mqh>    // for python analysis
//#import "shell32.dll"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
// Write File
//--TERMINAL_DATA_PATH-- C:\Program Files\MofidTrader\Tester\Agent-127.0.0.1-3000\MQL5\Files
// File place On live market C:\Program Files\MofidTrader\MQL5\Files  
// tavajoh: vase expert haye ham zaman, name avaz shavad

//#define numb_symbols 9
#define EXPERT_MAGIC 123456   // MagicNumber of the expert

#include <Trade\Trade.mqh>                  //include the library for execution of trades
#include <Trade\PositionInfo.mqh>           //include the library for obtaining information on positions
#include <Files\FileTxt.mqh>
#include <Trade\SymbolInfo.mqh>
CTrade            m_Trade;                  //structure for execution of trades
CPositionInfo     m_Position;               //structure for obtaining information of positions
CSymbolInfo       m_symbol;                 // symbol info object  ////**** kheyli khooobe, etelaat ziadi az namad be sadegi be ma mide
CFileTxt          m_file_txt;

bool First_tick=true;

//input ENUM_TIMEFRAMES period=PERIOD_H1;  // time frame
ENUM_TIMEFRAMES period;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum ENUM_INPUT_SYMBOLS
  {
   INPUT_SYMBOLS_CURRENT=0,   // current symbol
   INPUT_SYMBOLS_FILE=1,      // text file
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum ENUM_INPUT_SYMBOLS_PARAM
  {
   INPUT_SYMBOLS_CURRENT_PARAM=0,   // current symbol
   INPUT_SYMBOLS_FILE_PARAM=1,      // text file
  };

input ENUM_INPUT_SYMBOLS   InpInputSymbol=INPUT_SYMBOLS_FILE;   // works on ...
input ENUM_INPUT_SYMBOLS_PARAM   InpInputSymbol_param=INPUT_SYMBOLS_FILE_PARAM;   // works on ...

int sell_numb=0;
int tick_numb=0;

input int                  fast_ema_period=12;        // period of fast ma 
input int                  slow_ema_period=26;        // period of slow ma 
input int                  signal_period=9;           // period of averaging of difference 
input ENUM_APPLIED_PRICE   applied_price=PRICE_CLOSE; // type of price   

int      EA_Magic=12345;

input int                  fast_ma_period=3;             // period of fast ma 
input int                  slow_ma_period=10;            // period of slow ma 
input ENUM_MA_METHOD       ma_method=MODE_EMA;           // type of smoothing 

input int ExtDepth=5;              //ZigZag parameter asli 12/ man 5
input int ExtDeviation=5;          //ZigZag parameter asli 5/ man 5
input int ExtBackstep=3;           //ZigZag parameter asli 3/ man 3
//----------------zigzag2
input int ExtDepth2=12;              //ZigZag parameter asli 12/ man 5
input int ExtDeviation2=5;          //ZigZag parameter asli 5/ man 5
input int ExtBackstep2=3;           //ZigZag parameter asli 3/ man 3
//------------------------
input int                  Kperiod=5;                 // the K period (the number of bars for calculation) 
input int                  Dperiod=3;                 // the D period (the period of primary smoothing) 
input int                  slowing=3;                 // period of final smoothing       
input ENUM_STO_PRICE       price_field=STO_LOWHIGH;   // method of calculation of the Stochastic 

//+----------------------------------------------+
//|  Indicator osc input parameters                  |
//+----------------------------------------------+
input uint   length=15;
input uint   t3=3;
input double b=0.7;
//-------------------------
/*
      input string InpCommand="C:\\Program Files\\R\\R-3.6.0\\bin\\x64\\Rterm.exe";//Path to Rterm.exe
      input int InpOrder=50;//Order//200
      int InpBack=300;//Back//1000
      input int InpAhead=20;//Ahead//20
      #include <R.mqh>

      double PredictBuffer[];
      long R;
      bool recalc;
      bool dll_allowed=MQLInfoInteger(MQL_DLLS_ALLOWED);
      double hist[];
      double pred[];
   */
//-------------------------
int SanatSize=39; // tanha adad vabaste be excel
double RSI[];
double SAR[];
double OBVBuffer[];
double MACDBuffer[];
double MACDSignal[];
double Sup_Buffer[];
double Res_Buffer[];

double PTL_trend[];
double PTL_arrowCol[];
double PTL_arrowVal[];
//-------------------- python
datetime PythonCreatedFileTime;
bool FirstCreatedPythonFile=false;
int count_M1_PythonFileCreated=0;

datetime PythonCreatedFileTime_BuyCheck ;
bool FirstCreatedPythonFile_BuyCheck = false;
int count_M1_PythonFileCreated_BuyCheck = 0;

#include <MyExpertsClasses\PredictionLoader.mqh>
string model_tags[] = {"15minBuy", "1hBuy", "2hBuy", "3hBuy", "4hBuy", "1DBuy"};
string data_folder = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";

// string data_folder = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";
////double deltaMACD1_shakhes=0;
////double deltaMACD2_shakhes=0;
////double deltaMACD3_shakhes=0;
//double deltaMACD_Sig1_shakhes=0;
//double deltaMACD_Sig2_shakhes=0;
////double deltaMACD_Sig3_shakhes=0;
//bool deltaMACD1_shakhes_check=false;

//double Chaikin_Buffer[];
double Candle_Buffer[];
//osc Buffers
double BuyBuffer[];
double SellBuffer[];
double IndBuffer[],SigBuffer[];


//osc2 Buffers
double stochBuff_OSC[];
double signalBuff_OSC[];

//--------------------------------All sup & ress and zigzag parameter
double ZigZag_Buffer[];
double ZigZag_Buffer_F[][2]; //Filter aval
double ZigZag_Buffer_F2[][2]; //Filter dovom
double Supp_Z;
double Ress_Z;
int Supp_Z_STR;
int Ress_Z_STR;
string Ravand_symbol[];
double Supp_Z_symbol[];
double Ress_Z_symbol[];
int Supp_Z_STR_symbol[];
int Ress_Z_STR_symbol[];
int n_bar_Change_ravand[];
bool   Signal_From_zigzag[];
bool   Strong_Signal_From_zigzag[];
string Ravand;
bool Signal_From_zigzag_shakhes;
double Chg_RavandInPrice1nd;
double Chg_RavandInPrice2nd;
double Chg_RavandInPrice3nd;
double Chg_RavandInPrice4nd;
double Chg_RavandBarNumb1nd;
double Chg_RavandBarNumb2nd;
double Chg_RavandBarNumb3nd;
double Chg_RavandBarNumb4nd;
double shib_ravand_Now;
double shib_ravand1;
double shib_ravand2;
double shib_ravand3;
int    counter_Chng_Percent[];
int counter_Chng_Percent_shakhes;
double LastPercent[];
//-----------------------------All sup & ress and zigzag_Zstr parameter
double ZigZag_Buffer_Zstr[];
double ZigZag_Buffer_F_Zstr[][2]; //Filter aval
double shib_ravand_Zstr[];
double shib_ravand[];
double ZigZag_Buffer_F2_Zstr[][2]; //Filter dovom
double Supp_Z_Zstr;
double Ress_Z_Zstr;
int Supp_Z_STR_Zstr;
int Ress_Z_STR_Zstr;
string Ravand_symbol_Zstr[];
double Supp_Z_symbol_Zstr[];
double Ress_Z_symbol_Zstr[];
int Supp_Z_STR_symbol_Zstr[];
int Ress_Z_STR_symbol_Zstr[];
int n_bar_Change_ravand_Zstr[];
bool   Signal_From_zigzag_Zstr[];
bool   Strong_Signal_From_zigzag_Zstr[];
string Ravand_Zstr;
bool Signal_From_zigzag_shakhes_Zstr;
double Chg_RavandInPrice1nd_Zstr;
double Chg_RavandInPrice2nd_Zstr;
double Chg_RavandInPrice3nd_Zstr;
double Chg_RavandInPrice4nd_Zstr;
double Chg_RavandBarNumb1nd_Zstr;
double Chg_RavandBarNumb2nd_Zstr;
double Chg_RavandBarNumb3nd_Zstr;
double Chg_RavandBarNumb4nd_Zstr;
double shib_ravand_Now_Zstr;
double shib_ravand1_Zstr;
double shib_ravand2_Zstr;
double shib_ravand3_Zstr;
int    counter_Chng_Percent_Zstr[];
double LastPercent_Zstr[];

////saf kharid va foroush
//bool safKharid[];
//int count_safKharid_day[];
//string last_safKharid_start[];
//string last_safKharid_break[];
//bool complete_safe_kharid[];
//int safKharid_Start_H[];
//int safKharid_Start_m[];
//
//bool safFrush[];
//int count_safFrush_day[];
//string last_safFrush_start[];
//string last_safFrush_break[];
//---------------------------
string symbol_des[];
string symbol_path[];
int counter_array[];
int h;
double LastPrice_shakhes;
int symbol_size;

//int      handle_iCustom_Supp_ressist;
string m_symbols_array[];                  // array of symbol names
int    m_handles_array[][5][5];               // array of handles
int    m_handles_array2[][5][5];               // array of handles2
string m_symbols_array_param[][17];
string m_symbols_array_Portfolio[];
//double percent_Price_CH_sanat[75][2];
//double Hajm_Mabna[];
//int    Sanaat[];
//double Ave_Cost_3M[];
//double eps[];
//double P_on_E[];
//double ALL_number_Of_Sahm[];
//double DayNum_NoDeal_at_3M[];
//double DayNum_Open_at_3M[];
//double Ave_haghigh_Buy_3M[];
//double Rotbe_haghigh_Buy_3M[];
//double Ave_hoghoogh_Buy_3M[];
//double Rotbe_hoghoogh_Buy_3M[];
//double Ave_haghigh_Sell_3M[];
//double Rotbe_haghigh_Sell_3M[];
//double Ave_hoghoogh_Sell_3M[];
//double Rotbe_hoghoogh_Sell_3M[];

double this_day_CP[];
double Last_day_CP_Symbol[];
double WVP[];

double Ave_100_vol[];
double Ave_100_Price[];
double Ave_100_Price_Perc[];
double cost_of_deals[][2];
string Symbol_Last_sell_Time[][2];
string Symbol_Last_sell_Type[][2];
string Symbol_Last_Buy_Time[][2];
string Symbol_Last_Buy_Type[][2];

string DayBef_Buy_parameters[];
string BefBuy_signal[];
string Last_Day_Symbol_parameters[];
string Email_Buy_Parameters[]; //for write Email Buy Text 
string P3_Bazar_shakhes[];

int    Email_counter_sell[][4];
int    Email_counter_Buy[][4];
int    Email_counter_Deapth[];

int Count_shakhes_Email1=0;
int Count_shakhes_Email2=0;
//
//int   shakhesIndex=NULL;

double symbol_OPTION_STRIKE[];
double symbol_CONTRACT_SIZE[];
double symbol_ACCRUED_INTEREST[];
double symbol_FACE_VALUE[];
double symbol_LIQUIDITY_RATE[];
double symbol_SWAP_LONG[];
double symbol_SWAP_SHORT[];
double symbol_MARGIN_INITIAL[];
double symbol_MARGIN_SESSION_PRICE_LIMIT_MIN[];
double symbol_MARGIN_SESSION_PRICE_LIMIT_MAX[];
double symbol_MARGIN_HEDGED[];
string symbol_INFO_STRING[];
double percent_Price_Cl_day[];
double percent_Price_WVP_day[];
double percent_Price_now[];
double perc_Price_15day[];

double total_cost_sanat[];
double total_cost_sanat_dis[];   
double total_cost_disp=0;
//
//string Vazeiat_sanat[];
//bool vaz_sanat_cheaked[];

//double Posit_symb_cunt_sanat_dis[];
//double Nega_symb_cunt_sanat_dis[];
//double Zero_symb_cunt_sanat_dis[];
//double Up_mos3_symb_cunt_sanat_dis[];
//double Low_manf3_symb_sanat_dis[];
//double Posit_symb_cunt_sanat[];
//double Nega_symb_cunt_sanat[];
//double Zero_symb_cunt_sanat[];
//double Up_mos3_symb_cunt_sanat[];
//double Low_manf3_symb_sanat[];

double   WVPBef2B[][10];
datetime M1_400_BefBuyTime[];
double   WVPB2S[][26];
double   VolB2S[][26];
bool     SignalSellA[];
double   SignalSellA_price[];
double   SignalSellA_Loos[];
bool     SignalSellB[];
double   SignalSellB_price[];
double   SignalSellB_Loos[];
bool     SignalSellC[];
double   SignalSellC_price[];
double   SignalSellC_Loos[];
int      SignalSellAEmCunt[];
int      SignalSellBEmCunt[];
int      SignalSellCEmCunt[];
int   nearOfRess_EmCunt[];

string SymbolCandel[][5];
string SymbolCandelS[];
bool     shekast_Zigzag[];
double   LastZigzag[][2];
double   Pof_shekast_zigzag[];
//bool SanatIsZero[];
bool Sell_signal_Aft4H_MaxPBef3H[];
bool Sell_signal_after_2_h_shekast_max_of_loos[];
int Email_count_Warning_afterBuy[][27];

double     Sell_signal_after_2_h_shekast_max_of_loos_Percent[];
double     percent_shekast_zigzag[];
double     Sell_signal_Aft4H_MaxPBef3H_Percent[];
int        Sell_signal_after_2_h_shekast_max_of_loos_Hour[];
int        SignalSellA_Hour[];
int        SignalSellB_Hour[];
int        SignalSellC_Hour[];
int        Sell_signal_Aft4H_MaxPBef3H_Hour[];
int        shekast_Zigzag_Hour[];

int   Posit_symb_cunt=0;
int   Nega_symb_cunt=0;
int   Zero_symb_cunt=0;
int   Up_mos3_symb_cunt=0;
int   Low_manf3_symb_cunt=0;

int   total_Current_BUY=0;
int   total_Current_SELL=0;

int   Posit_symb_cunt_disp=0;
int   Nega_symb_cunt_disp=0;
int   Zero_symb_cunt_disp=0;
int   Up_mos3_symb_cunt_disp=0;
int   Low_manf3_symb_cunt_disp=0;
   
   
double MaxOfProfit_Position[];
int H_MaxOfProfit_FromBuy[];
double MaxOfLoos_Position[];
int H_MaxOfLoos_FromBuy[];
bool writedailydata=false;
bool first_open_file1=true;
bool first_open_file2=true;
bool first_open_file3=true;
//string   m_symbols_Sanat_array[];
//double   Bar_high_sanat_Array[][3];
//double   Bar_high_noshadow_sanat_Array[][3];
//double   LastD_Bar_high_sanat_Array[];
//double   LastD_Bar_high_noshadow_sanat_Array[];
//double   AveShakhes_sanatDay_Array[];
//double   sanat_to_Sup_perc_Array[];
//double   sanat_to_Res_perc_Array[];
//string   Ravand_sanat_Array[];

bool     vagarayiMosbat[];
bool     vagarayiManfi[];

int  Last_notif_time=10;

double   lastVol[];
double   Vol_win_sell[];
double   Vol_win_buy[];
int      tick_cunt_win_buy[];
int      tick_cunt_win_sell[];
double LastTickAsk[];
double LastTickBid[];
double Sum_Perc_P_win_sell[];
double Ave_Perc_P_win_sell[];
double Sum_Perc_P_win_buy[];
double Ave_Perc_P_win_buy[];

datetime position_time;

string Buy_Strategy[];
string Sell_Strategy[];

int day_buy_c=0;
bool today_buy_done=false;

bool signalVol[];
int signalVol_nBar[];
string Candle_type;
double deltaRSI;
int handle_rsi2,handle_MACD2,handle_OBV2,handle_Sup_resis2,handle_forecastosc2
    ,handle_ZigZag2,handle_ZigZag3,handle_Candle2,handle_iStoch,handle_Trend1;

input ENUM_APPLIED_VOLUME  volume=VOLUME_REAL;                // volume used 

int file_handle;
int file_handle_2;
int file_handle_3;
int file_handle_4;
string STring_Last_call_time;
// not buy and sell in one bar parameter 
struct Datetime
  {
   datetime          time[];
  };

Datetime lastbar_time[];
datetime new_bar[];

string date_of_accept_buy[];
bool tommorow_Buy[];
bool tommorow_sell[];
bool Write_his_buy[];

bool tommorow_Buy_dailyAlert[];
bool tommorow_sell_dailyAlert[];

string Buy_Strategy_dailyAlert[];
int today_buy=0;

string Posit_signals[];
int numb_Posit_sig[];
string Nega_signals[];
int numb_Nega_sig[];

//bool Saf_Kharid_Sig[];
//bool shekast_Saf_Froush_Sig[];
//bool shekast_Saf_Kharid_Sig[];
//bool Saf_Froush_Sig[];
bool signalVol_neg_Sig[];
bool signalVol_pos_Sig[];
bool Chng_Percent_Sig[];
bool Clpric_to_lastP_pos_Sig[];
bool Clpric_to_lastP_neg_Sig[];
bool High_Bar_Manfi_Sig[];
bool High_Bar_Mosbat_Sig[];
bool near_sup_Sig[];
bool near_ress_Sig[];
bool more_13_eslah_Sig[];
bool LastRav_Sig[];
bool MACD_pos_Sig[];
bool Shekast_OSC_mosbat_Sig[];
bool Shekast_OSC_manfi_Sig[];
bool OSC_zir25_Soodi_Sig[];

double avr_numb_Nega_sig=0;
double avr_numb_Posit_sig=0;

int Perc20d_upper30perc_cunt=0;
int Perc20d_upper20perc_cunt=0;
int Perc20d_upper10perc_cunt=0;
int Perc20d_upper6perc_cunt=0;
int Perc20d_lower6perc_cunt=0;
int Perc20d_lowerneg6perc_cunt=0;
int Perc20d_lowerneg10perc_cunt=0;
int Perc20d_lowerneg20perc_cunt=0;
string Day_Analysis_for_15day;
string LastTimeWritefile3="none";

double shakhes_to_Sup_perc=0;
double shakhes_to_Res_perc=0;
double shakhes_perc=0;
double shakhes_Hamvazn_perc=0;

int rotbe_naghd[];

double Ave_R1_Disp=0;
double Ave_R2_Disp=0;
double Ave_R3_Disp=0;
double Ave_R4_Disp=0;
double Ave_R5_Disp=0;
double Ave_R6_Disp=0;
double Ave_R7_Disp=0;
double Ave_R8_Disp=0;

int numb_growD_5day[];
int numb_growD_10day[];
int numb_growD_20day[];
//------------------- for timeframe 
double Supp_Z_TF[4];
double Ress_Z_TF[4];
double Supp_Z_STR_TF[4];
double Ress_Z_STR_TF[4];
string Ravand_TF[4];
string Ravand_Zstr_TF[4];
int n_bar_Change_ravand_TF[4];
int n_bar_Change_ravand_Zstr_TF[4];
double shib_ravand_Now_TF[4];
double shib_ravand1_TF[4];
double shib_ravand2_TF[4];
double shib_ravand3_TF[4];
double shib_ravand_Now_Zstr_TF[4];
double shib_ravand1_Zstr_TF[4];
double shib_ravand2_Zstr_TF[4];
double shib_ravand3_Zstr_TF[4];

double now_ravand_perc_TF[4];
double sec_ravand_perc_TF[4];
double third_ravand_perc_TF[4];

//indicators
double stochBuff_OSC_TF[10][4];
double signalBuff_OSC_TF[10][4];
double RSI_TF[10][4];
double SAR_TF[10][4];
double MACDBuffer_TF[10][4];
double MACDSignal_TF[10][4];


double deltaMACD1_TF[4];
double deltaMACD2_TF[4];
double deltaMACD3_TF[4];
double deltaMACD_Sig1_TF[4];
double deltaMACD_Sig2_TF[4];
double deltaMACD_Sig3_TF[4];
double shibMACD1_TF[4];
double shibMACD2_TF[4];
double LastN_MACD_chng_TF[4];

string RSI_Status[4];	  
string MACD_Status[4];
string Stochastic_Status[4];	
string PTL_signal[4];	
string PTL_Status[4];	


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

//--- Set event generation frequency
//   EventSetTimer(60); // 1 h second

//--- initial chart indicator add
   handle_Sup_resis2=iSAR(Symbol(),Period(),0.02,0.2);
   ChartIndicatorAdd(0,0,handle_Sup_resis2);
   handle_rsi2=iRSI(Symbol(),Period(),14,applied_price);
   ChartIndicatorAdd(0,0,handle_rsi2);
   handle_MACD2=iMACD(Symbol(),Period(),fast_ema_period,slow_ema_period,signal_period,applied_price);
   ChartIndicatorAdd(0,0,handle_MACD2);
   handle_OBV2=iOBV(Symbol(),Period(),volume);
   ChartIndicatorAdd(0,0,handle_OBV2);
   
   handle_forecastosc2=iCustom(Symbol(),Period(),"forecastoscilator",length,t3,b,applied_price);
   ChartIndicatorAdd(0,0,handle_forecastosc2);
   handle_ZigZag2=iCustom(Symbol(),Period(),"ZigZag",ExtDepth,ExtDeviation,ExtBackstep);//
   ChartIndicatorAdd(0,0,handle_ZigZag2);
   handle_ZigZag3=iCustom(Symbol(),Period(),"ZigZag",ExtDepth2,ExtDeviation2,ExtBackstep2);//
   ChartIndicatorAdd(0,0,handle_ZigZag3);
   
   handle_iStoch=iStochastic(Symbol(),Period(),Kperiod,Dperiod,slowing,ma_method,price_field); 
   ChartIndicatorAdd(0,0,handle_iStoch);

   handle_Trend1=iCustom(Symbol(),Period(),"PTL");//
   ChartIndicatorAdd(0,0,handle_Trend1);
   
   
//handle_Candle2=iCustom(Symbol(),Period(),"MACD_Divergence");//
//ChartIndicatorAdd(0,0,handle_Candle2);

//-----Read Symbols and parameter from File
// File place On live market C:\Program Files\MofidTrader\MQL5\Files
// new: C:\Users\hossein\AppData\Roaming\MetaQuotes\Terminal\2506E8E7E4116548D478CE2C3598FAB1\MQL5\Files
// on tester C:\Program Files\MofidTrader\Tester\Agent-127.0.0.1-3000\MQL5\Files
// C:\Users\hossein\AppData\Roaming\MetaQuotes\Tester\2506E8E7E4116548D478CE2C3598FAB1\Agent-127.0.0.1-3000\MQL5\Files
//int size=ArraySize(m_symbols_array);

//--------- file my_portfolio read
   //string terminal_data_path=TerminalInfoString(TERMINAL_DATA_PATH); 
   //string My_Portfolio=terminal_data_path+"\\MQL5\\"+"My_Portfolio.csv"; 
   //string FULL_3_99=terminal_data_path+"\\MQL5\\"+"FULL_3_99.csv"; 
   
   int Portfolio_inp=FileOpen("My_Portfolio.csv",FILE_READ|FILE_CSV);
   if(Portfolio_inp==INVALID_HANDLE)
     {
      Alert("Error opening file My_Portfolio ");
      return(INIT_FAILED);
     }
   Print("=== Start read Portfolio_inp parameter ===");
   int i=0;
   while(!FileIsEnding(Portfolio_inp))
     {
         string str=FileReadString(Portfolio_inp);
         //Print("str",str);
         ArrayResize(m_symbols_array_Portfolio,i+1,10);
         m_symbols_array_Portfolio[i]=str;
         // ArrayPrint(m_symbols_array_param);
      i++;
     }

   Print("parameter of Portfolio_inp is write successful");
   ArrayPrint(m_symbols_array_Portfolio);
   FileClose(Portfolio_inp);
   int size_Portfolio=ArraySize(m_symbols_array_Portfolio);
//---
//-------- file my_portfolio read
   int param_h=FileOpen("FULL_3_99.csv",FILE_READ|FILE_CSV);
   if(param_h==INVALID_HANDLE)
     {
      Alert("Error opening file FULL_3_99 ");
      return(INIT_FAILED);
     }

   Print("=== Start read parameter ===");
   i=0;
   while(!FileIsEnding(param_h))
     {
      for(int j=0; j<1; j++)
        {
         string str=FileReadString(param_h);
         //Print("str",str);

         ArrayResize(m_symbols_array_param,i+1,10);
         m_symbols_array_param[i][j]=str;
         // ArrayPrint(m_symbols_array_param);

         if(j==6 && FileIsLineEnding(param_h)==false)
           {
            Alert("file have more column than 7   !!!");
            FileClose(param_h);
            return(INIT_FAILED);
           }
        }
      i++;
     }

   Print("parameter of symbol is write successful");
   ArrayPrint(m_symbols_array_param);
   FileClose(param_h);
//---

//----------------------------- parameter Array Define--------------------
   int size=ArrayRange(m_symbols_array_param,0);


   int h=size;
   for(int i=0; i<size; i++)
     {
      if(!m_symbol.Name(m_symbols_array_param[i][0]))
        {
         h=h-1;
         Alert("Market dont have or not copy Symbol : ",m_symbols_array_param[i][0]);
         Print("Market dont have or not copy Symbol : ",m_symbols_array_param[i][0]);

        }
      }

   ArrayResize(m_symbols_array,h,10);
         
   size=ArrayRange(m_symbols_array,0);


   Print("size",size);
   int counter=0;
   for(int i=0; i<size; i++)
     {
      counter++;

      //symbol names
      m_symbols_array[i]=m_symbols_array_param[i][0];


      //Print("Iteration ",counter,"   ",m_symbols_array[i],"parameter def Success");
     }
//---------------------Create Indicator Handels-----------------------------
   size=ArraySize(m_symbols_array);
   //ArrayResize(m_symbols_array,10);
   //m_symbols_array[size]="شاخص کل6";
   //size=ArraySize(m_symbols_array);
   //ArrayPrint(m_symbols_array);
   ArrayResize(m_handles_array,size,10);
   ArrayResize(m_handles_array2,size,10);
   symbol_size=size; // sanat numb + 1 (shakhes kol)

   //shakhesIndex=(size-1);

   for(int i=0;i<size;i++)
     {
     for(int t=0;t<4;t++)
       {
//string period=PERIOD_H1;
//--------- t=0 : timeframe : 15 min
//--------- t=1 : timeframe : h
//--------- t=2 : timeframe : H 4
//--------- t=3 : timeframe : D1


if(t==0)
  {
    period    = PERIOD_M15; }
  else if(t==1)  
  {
    period    = PERIOD_H1; }
  else if(t==2)
  {
    period    = PERIOD_H4; }
  else        
  { period    = PERIOD_D1; }





      int rsi,MACD,OBV,Sup_Res,Fcastosci,ZigZag,ZigZag_2,Candle,Stoch,PTL;
      if(!CreateHandles(m_symbols_array[i],period,rsi,MACD,OBV,Sup_Res,Fcastosci,ZigZag,ZigZag_2,Candle,Stoch,PTL))
         continue; //return(INIT_FAILED);

      m_handles_array[i][0][t]=rsi;
      m_handles_array[i][1][t]=MACD;
      m_handles_array[i][2][t]=OBV;
      m_handles_array[i][3][t]=Sup_Res;
      m_handles_array[i][4][t]=Candle;

      m_handles_array2[i][0][t]=Fcastosci;
      m_handles_array2[i][1][t]=ZigZag;
      m_handles_array2[i][2][t]=ZigZag_2;
      m_handles_array2[i][3][t]=Stoch;
      m_handles_array2[i][4][t]=PTL;
      // ArrayResize(new_bar,i+1,10);
      // new_bar[size]=NULL;

       }

     }
     
     
//+++++++++++++ 
//++++++++======== Symbol Info ========+++++++++//
//+++++++++++++
   size=ArraySize(m_symbols_array);
   ArrayResize(symbol_des,size,10);
   ArrayResize(symbol_path,size,10);
   for(int i=0;i<size;i++)
     {
      symbol_des[i]=SymbolInfoString(m_symbols_array[i],SYMBOL_DESCRIPTION);
      symbol_path[i]=SymbolInfoString(m_symbols_array[i],SYMBOL_PATH);
     }
     
     
     
//--------------------resize arrays to symbol numbs----------------
   size=ArraySize(m_symbols_array);
   
ArrayResize(P3_Bazar_shakhes,size,10);

//------------------
   ArrayResize(Signal_From_zigzag,size,10);
   ArrayResize(Strong_Signal_From_zigzag,size,10);
   ArrayResize(n_bar_Change_ravand,size,10);
   ArrayResize(Ravand_symbol,size,10);
   ArrayResize(Supp_Z_symbol,size,10);
   ArrayResize(Ress_Z_symbol,size,10);
   ArrayResize(Supp_Z_STR_symbol,size,10);
   ArrayResize(Ress_Z_STR_symbol,size,10);
   ArrayResize(LastPercent,size,10);
   ArrayResize(counter_Chng_Percent,size,10);
//------------------------
   ArrayResize(Signal_From_zigzag_Zstr,size,10);
   ArrayResize(Strong_Signal_From_zigzag_Zstr,size,10);
   ArrayResize(n_bar_Change_ravand_Zstr,size,10);
   ArrayResize(Ravand_symbol_Zstr,size,10);
   ArrayResize(Supp_Z_symbol_Zstr,size,10);
   ArrayResize(Ress_Z_symbol_Zstr,size,10);
   ArrayResize(Supp_Z_STR_symbol_Zstr,size,10);
   ArrayResize(Ress_Z_STR_symbol_Zstr,size,10);
   ArrayResize(LastPercent_Zstr,size,10);
   ArrayResize(counter_Chng_Percent_Zstr,size,10);
//------------------------


   ArrayResize(cost_of_deals,size,10);
   ArrayResize(Symbol_Last_sell_Time,size,10);
   ArrayResize(Symbol_Last_sell_Type,size,10);
   ArrayResize(Symbol_Last_Buy_Time,size,10);
   ArrayResize(Symbol_Last_Buy_Type,size,10);
   
   ArrayResize(Ave_100_Price,size,10);
   ArrayResize(Ave_100_Price_Perc,size,10);
   ArrayResize(Ave_100_vol,size,10);
   ArrayResize(Email_counter_sell,size,10);
   ArrayResize(Email_counter_Buy,size,10);
   ArrayResize(Email_counter_Deapth,size,10);
   ArrayResize(DayBef_Buy_parameters,size,10);
   ArrayResize(Email_Buy_Parameters,size,10);
   ArrayResize(Last_Day_Symbol_parameters,size,10);

   ArrayResize(this_day_CP,size,10);
   ArrayResize(Last_day_CP_Symbol,size,10);
   ArrayResize(WVP,size,10);
   ArrayResize(percent_Price_Cl_day,size,10);
   ArrayResize(percent_Price_WVP_day,size,10);
   
   ArrayResize(percent_Price_now,size,10);
   ArrayResize(perc_Price_15day,size,10);
   ArrayResize(counter_array,size,10);

   ArrayResize(M1_400_BefBuyTime,size,10);

   ArrayResize(WVPB2S,size,10);
   ArrayResize(VolB2S,size,10);
   ArrayResize(SignalSellA,size,10);
   ArrayResize(SignalSellA_price,size,10);
   ArrayResize(SignalSellA_Loos,size,10);
   ArrayResize(SignalSellB,size,10);
   ArrayResize(SignalSellB_price,size,10);
   ArrayResize(SignalSellB_Loos,size,10);
   ArrayResize(SignalSellC,size,10);
   ArrayResize(SignalSellC_price,size,10);
   ArrayResize(SignalSellC_Loos,size,10);
   ArrayResize(WVPBef2B,size,10);
   ArrayResize(vagarayiMosbat,size,10);
   ArrayResize(vagarayiManfi,size,10);
   ArrayResize(SignalSellAEmCunt,size,10);
   ArrayResize(SignalSellBEmCunt,size,10);
   ArrayResize(SignalSellCEmCunt,size,10);
   ArrayResize(nearOfRess_EmCunt,size,10);

   ArrayResize(signalVol,size,10);
   ArrayResize(signalVol_nBar,size,10);
   ArrayResize(SymbolCandel,size,10);
   ArrayResize(SymbolCandelS,size,10);

   ArrayResize(shekast_Zigzag,size,10);
   ArrayResize(LastZigzag,size,10);
   ArrayResize(Pof_shekast_zigzag,size,10);

   ArrayResize(MaxOfProfit_Position,size,10);
   ArrayResize(H_MaxOfProfit_FromBuy,size,10);
   ArrayResize(MaxOfLoos_Position,size,10);
   ArrayResize(H_MaxOfLoos_FromBuy,size,10);
   ArrayResize(BefBuy_signal,size,10);

   ArrayResize(Email_count_Warning_afterBuy,size,10);
   ArrayResize(Sell_signal_Aft4H_MaxPBef3H,size,10);
   ArrayResize(Sell_signal_after_2_h_shekast_max_of_loos,size,10);

   ArrayResize(Sell_signal_after_2_h_shekast_max_of_loos_Percent,size,10);
   ArrayResize(percent_shekast_zigzag,size,10);
   ArrayResize(Sell_signal_Aft4H_MaxPBef3H_Percent,size,10);
   ArrayResize(Sell_signal_after_2_h_shekast_max_of_loos_Hour,size,10);
   ArrayResize(SignalSellA_Hour,size,10);
   ArrayResize(SignalSellB_Hour,size,10);
   ArrayResize(SignalSellC_Hour,size,10);
   ArrayResize(Sell_signal_Aft4H_MaxPBef3H_Hour,size,10);
   ArrayResize(shekast_Zigzag_Hour,size,10);



   ArrayResize(lastVol,size,10);
   ArrayResize(Vol_win_sell,size,10);
   ArrayResize(tick_cunt_win_sell,size,10);
   ArrayResize(Vol_win_buy,size,10);
   ArrayResize(tick_cunt_win_buy,size,10);
   ArrayResize(LastTickAsk,size,10);
   ArrayResize(LastTickBid,size,10);
   ArrayResize(Ave_Perc_P_win_buy,size,10);
   ArrayResize(Sum_Perc_P_win_buy,size,10);
   ArrayResize(Ave_Perc_P_win_sell,size,10);
   ArrayResize(Sum_Perc_P_win_sell,size,10);
//-----------
   ArrayResize(date_of_accept_buy,size,10);
   ArrayResize(tommorow_Buy,size,10);
   ArrayResize(tommorow_sell,size,10);
   ArrayResize(Write_his_buy,size,10);
   
   
   ArrayResize(Buy_Strategy,size,10);
   ArrayResize(Sell_Strategy,size,10);
   
   ArrayResize(tommorow_Buy_dailyAlert,size,10);
   ArrayResize(tommorow_sell_dailyAlert,size,10);
   
   ArrayResize(Buy_Strategy_dailyAlert,size,10);
//------------

   
   ArrayResize(Posit_signals,size,10);
   ArrayResize(numb_Posit_sig,size,10);
   ArrayResize(Nega_signals,size,10);
   ArrayResize(numb_Nega_sig,size,10);
   
   //ArrayResize(Saf_Kharid_Sig,size,10);
   //ArrayResize(shekast_Saf_Froush_Sig,size,10);
   //ArrayResize(shekast_Saf_Kharid_Sig,size,10);
   //ArrayResize(Saf_Froush_Sig,size,10);
   ArrayResize(signalVol_neg_Sig,size,10);
   ArrayResize(signalVol_pos_Sig,size,10);
   ArrayResize(Chng_Percent_Sig,size,10);
   ArrayResize(Clpric_to_lastP_pos_Sig,size,10);
   ArrayResize(Clpric_to_lastP_neg_Sig,size,10);
   ArrayResize(High_Bar_Manfi_Sig,size,10);
   ArrayResize(High_Bar_Mosbat_Sig,size,10);
   ArrayResize(near_sup_Sig,size,10);
   ArrayResize(near_ress_Sig,size,10);
   ArrayResize(more_13_eslah_Sig,size,10);
   ArrayResize(LastRav_Sig,size,10);
   ArrayResize(MACD_pos_Sig,size,10);
   ArrayResize(Shekast_OSC_mosbat_Sig,size,10);
   ArrayResize(Shekast_OSC_manfi_Sig,size,10);
   ArrayResize(OSC_zir25_Soodi_Sig,size,10);
   
   //ArrayResize(rotbe_naghd,size,10);
   ArrayResize(numb_growD_5day,size,10);
   ArrayResize(numb_growD_10day,size,10);
   ArrayResize(numb_growD_20day,size,10);


//----------
   for(int i=0;i<size;i++) // set array to Zero
     {
      date_of_accept_buy[i]=0;
      tommorow_Buy[i]=false;
      tommorow_sell[i]=false;
      Write_his_buy[i]=false;
      

      counter_array[i]=0;

      M1_400_BefBuyTime[i]=NULL;

      this_day_CP[i]=0;
      Last_day_CP_Symbol[i]=0;
      percent_Price_Cl_day[i]=0;
      percent_Price_WVP_day[i]=0;
      percent_Price_now[i]=0;
      perc_Price_15day[i]=0;
      
      WVP[i]=0;
      LastPercent[i]=0;
      counter_Chng_Percent[i]=0;

      Ravand_symbol[i]="";
      Supp_Z_symbol[i]=0;
      Ress_Z_symbol[i]=0;
      Supp_Z_STR_symbol[i]=0;
      Ress_Z_STR_symbol[i]=0;


      vagarayiMosbat[i]=false;
      vagarayiManfi[i]=false;
      signalVol[i]=false;
      signalVol_nBar[i]=0;

      //Sell_signal_Aft4H_MaxPBef3H[i]=false;
      //Sell_signal_after_2_h_shekast_max_of_loos[i]=false;

      shekast_Zigzag[i]=false;
      LastZigzag[i][0]=0;
      LastZigzag[i][1]=0;
      Pof_shekast_zigzag[i]=0;

      Signal_From_zigzag[i]=false;
      Strong_Signal_From_zigzag[i]=false;
      Strong_Signal_From_zigzag_Zstr[i]=false;

      Sell_signal_after_2_h_shekast_max_of_loos_Percent[i]=0;
      percent_shekast_zigzag[i]=0;
      Sell_signal_Aft4H_MaxPBef3H_Percent[i]=0;
      Sell_signal_after_2_h_shekast_max_of_loos_Hour[i]=0;
      SignalSellA_Hour[i]=0;
      SignalSellB_Hour[i]=0;
      SignalSellC_Hour[i]=0;
      Sell_signal_Aft4H_MaxPBef3H_Hour[i]=0;
      shekast_Zigzag_Hour[i]=0;

      MaxOfProfit_Position[i]=0;
      H_MaxOfProfit_FromBuy[i]=0;
      MaxOfLoos_Position[i]=0;
      H_MaxOfLoos_FromBuy[i]=0;

      lastVol[i]=0;
      Vol_win_sell[i]=0;
      Vol_win_buy[i]=0;
      tick_cunt_win_buy[i]=0;
      tick_cunt_win_sell[i]=0;
      LastTickAsk[i]=0;
      LastTickBid[i]=0;
      Sum_Perc_P_win_sell[i]=0;
      Ave_Perc_P_win_sell[i]=0;
      Sum_Perc_P_win_buy[i]=0;
      Ave_Perc_P_win_buy[i]=0;
      Buy_Strategy[i]="_";
      Sell_Strategy[i]="_";
      
     tommorow_Buy_dailyAlert[i]=false;
     tommorow_sell_dailyAlert[i]=false;
     Buy_Strategy_dailyAlert[i]="_";
	 
    Posit_signals[i]="Posit_signals: ";
	numb_Posit_sig[i]=0;
	Nega_signals[i]="Nega_signals: ";
	numb_Nega_sig[i]=0;
	 
 signalVol_neg_Sig[i]=false;
 signalVol_pos_Sig[i]=false;
 Chng_Percent_Sig[i]=false;
 Clpric_to_lastP_pos_Sig[i]=false;
 Clpric_to_lastP_neg_Sig[i]=false;
 High_Bar_Manfi_Sig[i]=false;
 High_Bar_Mosbat_Sig[i]=false;
 near_sup_Sig[i]=false;
 near_ress_Sig[i]=false;
 more_13_eslah_Sig[i]=false;
 LastRav_Sig[i]=false;
 MACD_pos_Sig[i]=false;
 Shekast_OSC_mosbat_Sig[i]=false;
 Shekast_OSC_manfi_Sig[i]=false;
 OSC_zir25_Soodi_Sig[i]=false;
 
  numb_growD_5day[i]=0;
  numb_growD_10day[i]=0;
  numb_growD_20day[i]=0;

     }

     
// NUN kardan hame data haye lazem

     
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//IndicatorRelease(ExtHandle);
   int size=ArraySize(m_symbols_array);
   for(int i=0;i<size;i++)
     {
     
          for(int t=0;t<4;t++)
       {
//--------- t=0 : timeframe : 15 min
//--------- t=1 : timeframe : h
//--------- t=2 : timeframe : H 4
//--------- t=3 : timeframe : D1

      IndicatorRelease(m_handles_array2[i][4][t]);
      IndicatorRelease(m_handles_array2[i][3][t]);
      IndicatorRelease(m_handles_array2[i][2][t]);
      IndicatorRelease(m_handles_array2[i][1][t]);
      IndicatorRelease(m_handles_array2[i][0][t]);

      IndicatorRelease(m_handles_array[i][4][t]);
      IndicatorRelease(m_handles_array[i][3][t]);
      IndicatorRelease(m_handles_array[i][2][t]);
      IndicatorRelease(m_handles_array[i][1][t]);
      IndicatorRelease(m_handles_array[i][0][t]);
      }
     }
// if(dll_allowed)
// RDeinit(R);
   Comment("");
//--- Termination of event generation
   EventKillTimer();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
//void OnTimer()
void OnTick()
  {
  
//----------------------------- Time Parameter -------------------------------//
   MqlDateTime CTi_Stru,BTi_Stru,Last_call_time;         // 2 struct for Current time and Bar Time
   datetime Ctime=TimeCurrent(CTi_Stru);                 // Current Time From Server(open market 9 to 12:30)

   PredictionLoader loader;
   loader.SetModelTags(model_tags);
   
   
  //Files
   FileDelete("DayParameter.csv");
   FileDelete("DayParameter_Portfolio.csv");
   //Print("hi im in timer" );
   tick_numb=tick_numb+1;         // shomare andaz tedad ejraye tabe OnTime ya OnTick
                                  //Print(tick_numb);
   file_handle=FileOpen("Check_Strategy_TESTER.csv",FILE_READ|FILE_WRITE|FILE_CSV); 
//   if(file_handle==INVALID_HANDLE) PrintFormat("Failed to open %s file, Error code = %d","Check_Strategy_TESTER.csv",GetLastError());
//    
   file_handle_2=FileOpen("DayParameter.csv",FILE_READ|FILE_WRITE|FILE_CSV);
   if(file_handle_2==INVALID_HANDLE) PrintFormat("Failed to open %s file, Error code = %d","DayParameter.csv",GetLastError());

   file_handle_3=FileOpen("15DayParameter.csv",FILE_READ|FILE_WRITE|FILE_CSV);
   //if(file_handle_3==INVALID_HANDLE) PrintFormat("Failed to open %s file, Error code = %d","Buy_history_TESTER.csv",GetLastError());

   file_handle_4=FileOpen("DayParameter_Portfolio.csv",FILE_READ|FILE_WRITE|FILE_CSV);
   if(file_handle_4==INVALID_HANDLE) PrintFormat("Failed to open %s file, Error code = %d","DayParameter_Portfolio.csv",GetLastError());
   


//-----------------------------python
    string outputFile = "prediction_result.txt";
    string pythonExe = "C:\\Users\\Hossein\\AppData\\Local\\Programs\\Python\\Python313\\python.exe";       // مسیر python.exe را به سیستم خودت تغییر بده
    string pythonScript = "C:\\pythonFiles\\my5_tested_AND_OKfor_MQL5.py"; // مسیر اسکریپت پایتون را تغییر بده
    
//    FileDelete("inputFile_Python.csv"); // har tick bayad dobare sakhte beshe
//    int file_handle_inputPython = FileOpen("inputFile_Python.csv", FILE_WRITE|FILE_CSV|FILE_ANSI);
//    if(file_handle_inputPython != INVALID_HANDLE) PrintFormat("Failed to open %s file, Error code = %d","inputFile_Python.csv",GetLastError());



//==========================================
   first_open_file1=true;         // its for clear of Day Parameter File before each tick
                                  // For on check All Symbol in each Tick
   int size_Portfolio=ArraySize(m_symbols_array_Portfolio);
   int size=ArraySize(m_symbols_array);
   double total_cost=0;
   

        
double sum_numb_Posit_sig=0;
double sum_numb_Nega_sig=0;
int checked_symbol_count=0;
//------------------------------------- START
   for(int i=0;i<size;i++)
     {
        //Print("bazar is close","CTi_Stru.hour   ",CTi_Stru.hour);
        //if(CTi_Stru.hour<10 || CTi_Stru.hour>=13) //
        //{
        //continue;
        //}
        //if(CTi_Stru.hour==11) // 
        //{
        //continue;
        //}
        
        
      //--- Define some MQL5 Structures we will use for our trade
      MqlTick latest_price;      // To be used for getting recent/latest price quotes
      MqlTradeRequest mrequest;  // To be used for sending our trade requests
      MqlTradeResult mresult;    // To be used to get our trade results
      MqlRates mrate[];          // To be used to store the prices, volumes and spread of each bar
      MqlRates mrate_min[];      // To be used to store the prices, volumes and spread of each bar in min TimeFram
      MqlRates mrate_H1[];
      MqlRates mrate_HH1[];
      ZeroMemory(mrequest);      // Initialization of mrequest structure
      SymbolInfoTick(m_symbols_array[i],latest_price); //daryaf etelaat gheymati namad, dar in lahze
      double last_price=latest_price.ask;
      
      
//-------------------------------------------- sakht satr aval file ha
string S1_symbol="symbol,Last Bar Time";
string S2_BuyStr=",Buy_Strategy_dailyAlert,tommorow_Buy_dailyAlert";
string S3_Bazar_shakhes=",Vazeiat bazar va shakhes"//+",symbolshakhes,time from shakhes,Ravand_symbol[shakhesIndex]"
     +",numb of positive symbol,numb of negative symbol,numb of zero symbol,numb of +3 symbol,numb of -3 symbol";
     //+",cost deal kol(miliard tom),shakhes_to_Sup_perc,shakhes_to_Res_perc"
     //+",Bar_high_noshadow_shakhes,LastD_Bar_high_noshadow_shakhes,count_Chng_Percent_shakhes,avr_numb_Posit_sig,avr_numb_Nega_sig";
string S4_SanatP=",Vazeiat sanat"//Vazeiat sanat
     +",symbolsanat name,time from sanat,Ravand_symbol[SanatIndex],ALL num sym sanat(maybe some close)"
     +",numb_pos_symbol_sanat,numb_neg_symbol_sanat,numb_zero_symbol_sanat,Up_mos3_symb_cunt_sanat_dis,Low_manf3_symb_sanat_dis"
     +",total_cost_sanat_dis(miliard tom),sanat_to_Sup_perc,sanat_to_Res_perc"
     +",Bar_high_noshadow_sanat,LastD_Bar_high_noshadow_sanat,Ave_100_Price_Perc[SanatIndex],AVE_percent_Price_CH_sanat,Vazeiat_sanat";
string S5_Sahm_bonyad=",Tot_cost_of_symb(BToman),P/E,EPS,Ave_Cost_3M(MilionToman),rotbe_naghd,Ave_Price_naghd";
string S6_Win_B_or_S=",Vol_win_sell[i],tick_cunt_win_sell[i],Ave_Perc_P_win_sell[i],Vol_win_buy[i],tick_cunt_win_buy[i],Ave_Perc_P_win_buy[i]";
string S7_Indicators=",RSI"
     +",stochBuff_OSC"
     +",signalBuff_OSC"
     +",stoch-signal"
     +",deltaMACD1"
     +",deltaMACD2"
     +",deltaMACD3";
     //+",IndBuffer[0](green),SigBuffer[1](red)"
     //+",last_buy_oscSig_day,last_sell_oscSig_day";
string S8_RavandP=",Ravand,n_bar_Change_ravand[i],Ravand_zstr,n_bar_Change_ravand[i]_Zstr,"+"now_ravand_perc"
       +","+"sec_ravand_perc"+","+"third_ravand_perc";
string S9_Sup_and_RessP=",Bprice to Sup%,Bprice to Res%";
string S10_just_dayP=",percent_Price_now[i]"+","+"percent_Price_Cl_day"+",Bar high,Bar_high_noshadow"
     +",counter_Chng_Percent[i]"
     +",cost of deals in day(milion tom),Vol/Ave_100_vol";
string S101_saf_parameter=",safKharid"+","+"count_safKharid_day"+","+"last_safKharid_start"+","+"last_safKharid_break"
      +","+"safFrush"+","+"count_safFrush_day"+","+"last_safFrush_start"+","+"last_safFrush_break";
string S11_signals=",signalVol[i],signalVol_nBar[i],vagarayiManfi,vagarayiMosbat";
string S12_candels=",SymbolCandel0,SymbolCandel1,SymbolCandel2,SymbolCandel3,SymbolCandel4";
string S13_extraP=",Extra param"//Extra param
     +",Last_Price,Res_Price,Res_STR,Sup_Price,Sup_STR,Last_Souod_Or_Nozol_perc"
     +",Last Day Vol,this day Vol,Last Day TickN,this day TickN"
     +",99 day bar AV_pric,deltaRSI"
     +",Sup to Res% Bday_Zstr,Bprice to Sup%_Zstr"
     +",Bprice to Res%_Zstr,Res_Price_Zstr,Res_STR_Zstr,Sup_Price_Zstr,Sup_STR_Zstr,Last_Souod_Or_Nozol_perc_Zstr"
     +",Vol/mabna,this_day_CP,Last_day_CP_Symbol"
     +",cost day/Ave_Cost_3M"
     +",more description of symbol,more description of symbol";
     
string S_Neg_Sig=",Clpric_<2%BishtarAz_LastPric"+",shekast_Saf_Kharid"+",Saf_Froush"+",signalVol_neg"+",High_Bar_Manfi "
+",near_ress"+",Shekast_OSC_manfi ";

string S_Posit_Sig=",Saf_Kharid"+",shekast_Saf_Froush"+",signalVol_Pos"+",counter_Chng_Percent_more_5"+
",High_Bar_Mosbat "+",near_sup "+",more -13 eslah "+",LastRav<-17and now <8 "+",deltaMACD_mosbat "
+",OSC_zir25_Soodi "+",Shekast_OSC_mosbat "+",Clpric_<2%KamtarAz_LastPric";

string S14_FastCheck=S5_Sahm_bonyad+",numb_growD_5day[i]"+",numb_growD_10day[i]"+",numb_growD_20day[i]"+",percent_Price_CH_sanat"
+",Vazeiat_sanat"+",numb_Posit_sig,numb_Nega_sig+"+",Pos_Sigs :"+S_Posit_Sig+",Neg_Sigs :"+S_Neg_Sig;


string S15_ASK_to_bid_Perc=",S15_ASK_to_bid_Perc";


string day_parameter_first_row=S1_symbol
         +S2_BuyStr
         +S3_Bazar_shakhes
         //+S4_SanatP
         //+",Vazeiat saham"
         //+S5_Sahm_bonyad
         //+S6_Win_B_or_S
         +S7_Indicators
         +S8_RavandP
         +S9_Sup_and_RessP
         +S10_just_dayP
         //+S101_saf_parameter
         +S11_signals
         //+S12_candels
         //+S13_extraP
         //+",FastCheck"
         //+S14_FastCheck;
         +S15_ASK_to_bid_Perc;
         
      if(first_open_file1==true)
        {
         FileWrite(file_handle_2,day_parameter_first_row+"\r\n"+Last_Day_Symbol_parameters[i]);

         FileWrite(file_handle_4,day_parameter_first_row+"\r\n"+Last_Day_Symbol_parameters[i]);

         first_open_file1=false;
        }
        

      
      
//----------------------------------- TIMEFRAME For START
       for(int t=0;t<4;t++)
       {
//--------- t=0 : timeframe : 15 min
//--------- t=1 : timeframe : h
//--------- t=2 : timeframe : H 4
//--------- t=3 : timeframe : D1

if(t==0)
  {
    period    = PERIOD_M15; }
  else if(t==1)  
  {
    period    = PERIOD_H1; }
  else if(t==2)
  {
    period    = PERIOD_H4; }
  else        
  { period    = PERIOD_D1; }
  
  
      //--------------------------- open boodan bazar----------------
      //Print("CTi_Stru.hour",CTi_Stru.hour);
      if(CTi_Stru.hour==23) // dorost 9 boode
        {
         if(i<symbol_size)
           {
            // sefr kardan cont buy and sell param
            lastVol[i]=0;
            Vol_win_sell[i]=0;
            tick_cunt_win_sell[i]=0;
            Vol_win_buy[i]=0;
            tick_cunt_win_buy[i]=0;
            LastTickAsk[i]=0;
            LastTickBid[i]=0;
            Sum_Perc_P_win_sell[i]=0;
            Ave_Perc_P_win_sell[i]=0;
            Sum_Perc_P_win_buy[i]=0;
            Ave_Perc_P_win_buy[i]=0;
            
             
             today_buy_done=false;
             today_buy=0;
            
			 Posit_signals[i]="Posit_signals: ";
			 numb_Posit_sig[i]=0;
			 Nega_signals[i]="Nega_signals: ";
			 numb_Nega_sig[i]=0;
				 
			 //Saf_Kharid_Sig[i]=false;
			 //shekast_Saf_Froush_Sig[i]=false;
			 //shekast_Saf_Kharid_Sig[i]=false;
			 //Saf_Froush_Sig[i]=false;
			 signalVol_neg_Sig[i]=false;
			 signalVol_pos_Sig[i]=false;
			 Chng_Percent_Sig[i]=false;
			 Clpric_to_lastP_pos_Sig[i]=false;
			 Clpric_to_lastP_neg_Sig[i]=false;
			 High_Bar_Manfi_Sig[i]=false;
			 High_Bar_Mosbat_Sig[i]=false;
			 near_sup_Sig[i]=false;
			 near_ress_Sig[i]=false;
			 more_13_eslah_Sig[i]=false;
			 LastRav_Sig[i]=false;
			 MACD_pos_Sig[i]=false;
			 Shekast_OSC_mosbat_Sig[i]=false;
			 Shekast_OSC_manfi_Sig[i]=false;
			 OSC_zir25_Soodi_Sig[i]=false;
			 
          numb_growD_5day[i]=0;
          numb_growD_10day[i]=0;
          numb_growD_20day[i]=0;
           }
         // writedailydata=false;
         // counter_array[i]=0;
         //Print("bazar is close","CTi_Stru.hour   ",CTi_Stru.hour);
          //continue; //bazar is close + kolan bazar baste ro rad darim mikonim
        }


        
     //------------------------------------------------------------//
     // Print("Hi Im First of for ",m_symbols_array[i],
     // " tick_numb ",tick_numb," counter_array[i] ",counter_array[i]);
     //------------------------------------------------------------//
      
                   
                   







//---
               // min 
                  ArraySetAsSeries(mrate_HH1,true);
                  int copied2_HH1=CopyRates(m_symbols_array[i],PERIOD_H1,0,10,mrate_HH1); //Gets history data of MqlRates structure of a specified symbol-period in specified quantity into the rates_array array.
                  if(copied2_HH1>0)
                    { 
                          //Print("successful get history data for the symbol ,mrate_min ",m_symbols_array[i]); 
                    }
                    else
                    {
                     Print("Failed to get history data for the symbol ,mrate_min_sell "
                     ,"  ","CTi_Stru.hour =",CTi_Stru.hour,"   ",m_symbols_array[i]); 
                     Alert("cant calc Last_day_CP_symbol",m_symbols_array[i]+"   Last CP change to 0 but no analysis in this symbol "
                     +"Failed to get history data for the symbol ,mrate_min_sell "
                     );
                     continue;
                    }
      /////--------------------------------------
      // Print("Before MqlRates mrate  ALL Things For each Bar ",m_symbols_array[i]," tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      ///--------------------- MqlRates mrate  ALL Things For each Bar
      ////--------------------------

      //////Print("bef H1",i);
      ArraySetAsSeries(mrate_H1,true);
      int copied_H1=CopyRates(m_symbols_array[i],PERIOD_H1,0,30,mrate_H1); //Gets history data of MqlRates structure of a specified symbol-period in specified quantity into the rates_array array.
      if(copied_H1>0)
        {
        }
      else
        {
         Print("Failed to get history data for the symbol_copied_H1 ",m_symbols_array[i]);
         continue;
        }
      //Print("bef D1",i);
      ArraySetAsSeries(mrate,true);
      int copied=CopyRates(m_symbols_array[i],period,0,250,mrate); //Gets history data of MqlRates structure of a specified symbol-period in specified quantity into the rates_array array.
      if(copied>5)// arze avalie kam bar dare va momkene code ro kharab kone pas  felan ta 5 bar rad mikonam
        {
        }
      else
        {
         Print("Failed to get history data for the symbol ",m_symbols_array[i]);
         continue;
        }

      //Print("after copy",i);
      // mige: agar hanooz bar vase in rooz ijad nashode ya time ghabl az 9 hast ya hanooz moamele nadashtim ya 
      // kolan namad basteh ast,Alan kolan radesh mikonim in be depth market va tashkhis saf kharid o frosh sadameh mizaneh
      //Print("symbol",m_symbols_array[i],"mrate[0].time,TIME_DATE",TimeToString(mrate[0].time,TIME_DATE),"Ctime",TimeToString(Ctime,TIME_DATE));

      // inja daghighan new bar ra darim moshakhas mikonim
      //Print("mrate[0].time,TIME_DAT",TimeToString(mrate[0].time,TIME_DATE),"Ctime,TIME_DATE",TimeToString(Ctime,TIME_DATE));
      if(TimeToString(mrate[0].time,TIME_DATE)!=TimeToString(Ctime,TIME_DATE))
        {
         counter_array[i]=0;
         this_day_CP[i]=0; // gheymat payani emrooz in saham sefr sabt shavad be manaye basteh boodan
         counter_Chng_Percent[i]=0;
         counter_Chng_Percent_shakhes=0;
         LastPercent[i]=0;
         if(i<symbol_size)
           {
            // sefr kardan cont buy and sell param
            lastVol[i]=0;
            Vol_win_sell[i]=0;
            tick_cunt_win_sell[i]=0;
            Vol_win_buy[i]=0;
            tick_cunt_win_buy[i]=0;
            LastTickAsk[i]=0;
            LastTickBid[i]=0;
            Sum_Perc_P_win_sell[i]=0;
            Ave_Perc_P_win_sell[i]=0;
            Sum_Perc_P_win_buy[i]=0;
            Ave_Perc_P_win_buy[i]=0;
           }
         //writedailydata=false;
        // continue;
        }
      if(TimeToString(mrate[0].time,TIME_DATE)==TimeToString(Ctime,TIME_DATE)) // Alert new bar hast khodesh
        {
         counter_array[i]=counter_array[i]+1;
        
        }
        else
          {
         continue; // inja migim age sahm baste hast ya hanooz moamele naddarad rad beshe/ lazam hast vagarna dade rooz ghabl hesab mishe
          }

      //--------------------------------------
      //Print("Before Calc Last Close price of day in iran ",m_symbols_array[i]," tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
//      //--------------------Calc Last Close price of day in iran-------------------

            Last_day_CP_Symbol[i]=mrate[1].close;

      // -----------------------------------------------------------
      //Print("Before Bar Time Parameter ",m_symbols_array[i],
      //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      //-----------------------------Bar Time Parameter -------------------------------//
      TimeToStruct(mrate[0].time,BTi_Stru);   // Open Bar Time
      //----------------------------------------
      double cost_buy_day=0;
      double Tot_cost_of_symb=0;
      if(i<symbol_size)
        {
         //cost_of_deals[i][0]=mrate[0].tick_volume*((mrate[0].high+mrate[0].low)/2);
         cost_of_deals[i][1]=i;
         Symbol_Last_sell_Time[i][0]=m_symbols_array[i];
         Symbol_Last_Buy_Time[i][0]=m_symbols_array[i];
         cost_buy_day=(((mrate[0].low+mrate[0].high)/2)*mrate[0].tick_volume)/10000000; // arzesh moamele saham dar rooz ta alan be milion toman
         cost_of_deals[i][0]=(WVP[i]*mrate[0].tick_volume)/10000000000;
         total_cost=total_cost+(cost_of_deals[i][0]);
         //Tot_cost_of_symb=(ALL_number_Of_Sahm[i]*last_price)/10000000000;
         
         //total_cost_sanat[Sanaat[i]]=total_cost_sanat[Sanaat[i]]+(cost_of_deals[i][0]); 
        }
        
      //----------------------------------------
      // Print("Before 100 day bar price ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      //--------------------------- 200 day bar price 
      int bar_num=fmin(copied,100);
      if(counter_array[i]<5)
        {
         double Sum_100_Perc=0;
         double Sum_100_Price=0;
         double Sum_100_Vol=0;
         

         for(int ss=1; ss<bar_num; ss++)
           {
            if(mrate[ss].close==0)
              {
               continue;
              }
            else
              {
               Sum_100_Vol=Sum_100_Vol+mrate[ss].tick_volume;
               Sum_100_Perc=Sum_100_Perc+(((MathAbs(mrate[ss].open-mrate[ss].close))/mrate[ss].close)*100);
               Sum_100_Price=Sum_100_Price+((mrate[ss].open+mrate[ss].close)/2);

               Ave_100_Price[i]=(Sum_100_Price/ss);
               Ave_100_Price_Perc[i]=(Sum_100_Perc/ss);
               Ave_100_vol[i]=(Sum_100_Vol/ss);

               //Print("i",i,m_symbols_array[i]," ","ss",ss,"Sum_100",Sum_100,"Ave_100_Price[i][0]",Ave_100_Price[i][0] );
              }
           }
         //Print("hi+Im in 100 day bar price","tick_numb",tick_numb);
         //ArrayPrint(Ave_100_vol);
        }

      if(counter_array[i]<5 && i<symbol_size)
        {
         double Sum_100_Perc=0;
         double Sum_100_Price=0;
         double Sum_100_Vol=0;
         double Sum_100_cost=0; 
         
         numb_growD_5day[i]=0;
         numb_growD_10day[i]=0;
         numb_growD_20day[i]=0;
         
         for(int ss=1; ss<(bar_num-1); ss++)
           {
            if(mrate[ss].close==0)
              {
               continue;
              }
            else
              {
               Sum_100_Vol=Sum_100_Vol+mrate[ss].tick_volume;
               Sum_100_cost=Sum_100_cost+(((mrate[ss].open+mrate[ss].close)/2)*mrate[ss].tick_volume);
               
               Ave_100_vol[i]=(Sum_100_Vol/ss);
               //Ave_Cost_3M[i]=Sum_100_cost/ss;
               
               if(ss <= 5 && mrate[ss].close>mrate[ss+1].close)
                 {
                  numb_growD_5day[i]=numb_growD_5day[i]+1;
                 }
               if(ss <= 10 && mrate[ss].close>mrate[ss+1].close)
                 {
                  numb_growD_10day[i]=numb_growD_10day[i]+1;
                 }
               if(ss <= 20 && mrate[ss].close>mrate[ss+1].close)
                 {
                  numb_growD_20day[i]=numb_growD_20day[i]+1;
                 }
               //Print("i",i,m_symbols_array[i]," ","ss",ss,"Sum_100",Sum_100,"Ave_100_Price[i][0]",Ave_100_Price[i][0] );
              }
           }
         //Print("hi+Im in 100 day bar price","tick_numb",tick_numb);
         //ArrayPrint(Ave_100_vol);
        }


      //------------------------------------------------------------------------
      // Print("Before Indicators buffer Creation ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      //-----------------------------Indicators buffer Creation-----------------
      ArraySetAsSeries(RSI,true);                 // moratab kardan array bar hasb bar kandel
     ArraySetAsSeries(SAR,true);                 // moratab kardan array bar hasb bar kandel
      
      ArraySetAsSeries(MACDBuffer,true);    // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(MACDSignal,true);    // moratab kardan array bar hasb bar kandel
      
      //ArraySetAsSeries(OBVBuffer,true);     // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(Sup_Buffer,true);    // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(Res_Buffer,true);        // moratab kardan array bar hasb bar kandel
      
      ArraySetAsSeries(BuyBuffer,true);    // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(SellBuffer,true);    // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(IndBuffer,true);    // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(SigBuffer,true);    // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(ZigZag_Buffer,true);     // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(ZigZag_Buffer_Zstr,true);    // moratab kardan array bar hasb bar kandel
      //ArraySetAsSeries(Candle_Buffer,true);     // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(stochBuff_OSC,true);    // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(signalBuff_OSC,true);     // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(PTL_trend,true);                 // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(PTL_arrowCol,true);                 // moratab kardan array bar hasb bar kandel
      ArraySetAsSeries(PTL_arrowVal,true);                 // moratab kardan array bar hasb bar kandel
      
      
      if(CopyBuffer(m_handles_array[i][0][t],0,0,10,RSI)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores RSI buffer - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
        

// --------------------SAR
      if(CopyBuffer(m_handles_array[i][3][t],0,0,10,SAR)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores RSI buffer - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
        
      //-----MACD
      if(CopyBuffer(m_handles_array[i][1][t],0,0,10,MACDBuffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores MACDBuffer  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
        
      if(CopyBuffer(m_handles_array[i][1][t],0,1,10,MACDSignal)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores MACDSignal  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
      double deltaMACD1=MathArctan (MACDBuffer[0]-MACDBuffer[1]); //MathArctan returns a value within the range of -π/2 to π/2 radians
      double deltaMACD2=MathArctan(MACDBuffer[1]-MACDBuffer[2]);
      double deltaMACD3=MathArctan (MACDBuffer[2]-MACDBuffer[3]);
      
      double deltaMACD_Sig1=MathArctan (MACDSignal[0]-MACDSignal[1]);
      double deltaMACD_Sig2=MathArctan (MACDSignal[1]-MACDSignal[2]);
      double deltaMACD_Sig3=MathArctan (MACDSignal[2]-MACDSignal[3]);
      
        
      double shibMACD1=deltaMACD1-deltaMACD2;
      double shibMACD2=deltaMACD2-deltaMACD3;
      int    LastN_MACD_chng=0;

      for(int s=2;s<=9;s++)
        {
         if((MACDBuffer[s-1]-MACDBuffer[s])*(MACDBuffer[s-2]-MACDBuffer[s-1])<0)
           {
            LastN_MACD_chng=s-1;
            break;
           }
        }

      string MACDParameter=deltaMACD1+","+deltaMACD2+","+deltaMACD3
                           +","+shibMACD1+","+shibMACD2+","+LastN_MACD_chng;

      string MACDStringP="deltaMACD1,deltaMACD2,deltaMACD3,shibMACD1"
                         +",shibMACD2,LastN_MACD_chng";
                         
                     
      ////---------OBV
      //if(CopyBuffer(m_handles_array[i][2],0,0,5,OBVBuffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
      //  {
      //   if(tick_numb<3)
      //     {
      //      Print("Error copying indicatores OBVBuffer  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
      //      ResetLastError();
      //     }
      //   continue;
      //  }

        
//---------osc_ASLI
      if(CopyBuffer(m_handles_array2[i][3][t],0,0,10,stochBuff_OSC)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores stochBuff_OSC  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
        
      if(CopyBuffer(m_handles_array2[i][3][t],0,1,10,signalBuff_OSC)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores signalBuff_OSC  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }

// osc_custom 4 buffer
      if(CopyBuffer(m_handles_array2[i][0][t],0,0,5,IndBuffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores osc_Buffer1  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
      if(CopyBuffer(m_handles_array2[i][0][t],1,0,5,SigBuffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores osc_Buffer2  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
      if(CopyBuffer(m_handles_array2[i][0][t],2,0,5,SellBuffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores osc_Buffer3  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
        
      if(CopyBuffer(m_handles_array2[i][0][t],3,0,5,BuyBuffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores osc_Buffer4  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
        
        double last_buy_oscSig_day=0;
        double last_sell_oscSig_day=0; 
        
        for(int k=0;k<5;k++)
          {
           if(BuyBuffer[k]!=0)
             {
               last_buy_oscSig_day=k;
               break;
             }
           }
        for(int k=0;k<5;k++)
          {
           if(SellBuffer[k]!=0)
             {
               last_sell_oscSig_day=k;
               break;
             }
           }

//---
      if(CopyBuffer(m_handles_array2[i][1][t],0,0,copied,ZigZag_Buffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores ZigZag_Buffer  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }

      if(CopyBuffer(m_handles_array2[i][2][t],0,0,copied,ZigZag_Buffer_Zstr)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores ZigZag_Buffer  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
//PTL
// buffer 4

      if(CopyBuffer(m_handles_array2[i][4][t],4,0,copied,PTL_trend)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores PTL_1_Buffer  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
// buffer 8
      if(CopyBuffer(m_handles_array2[i][4][t],8,0,copied,PTL_arrowCol)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores PTL_1_Buffer  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }
        
// buffer 7
      if(CopyBuffer(m_handles_array2[i][4][t],7,0,copied,PTL_arrowVal)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<2)
           {
            Print("Error copying indicatores PTL_1_Buffer  - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }

// candle
      if(CopyBuffer(m_handles_array[i][4][t],4,0,50,Candle_Buffer)<0) /// rikhtan meghdar rsi dar har tick, dar array rsi
        {
         if(tick_numb<3)
           {
            Print("Error copying indicatores Candle_Buffer - error:",GetLastError(),"symbol",m_symbols_array[i],"tick_numb",tick_numb);
            ResetLastError();
           }
         continue;
        }


      Print(m_symbols_array[i],"Candle_Buffer[0]",Candle_Buffer[0]);
      ArrayPrint(Candle_Buffer);

      //----------------- set 5 last candle Type---------------------
      for(int j=0;j<5;j++)
        {
         if(Candle_Buffer[j]==10)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="Addi_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="Addi_Manfi";
              }
           }
         if(Candle_Buffer[j]==0)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="MARIBOZU_LONG_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="MARIBOZU_LONG_Manfi";
              }
           }
         if(Candle_Buffer[j]==1)
           {
            SymbolCandel[i][j]="DOJI";
           }
         if(Candle_Buffer[j]==2)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="SPINNING TOP_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="SPINNING TOP_Manfi";
              }
           }
         if(Candle_Buffer[j]==3)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="HAMMER_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="HAMMER_Manfi";
              }
           }
         if(Candle_Buffer[j]==4)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="TURN HAMMER_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="TURN HAMMER_Manfi";
              }
           }
         if(Candle_Buffer[j]==5)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="LONG_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="LONG_Manfi";
              }
           }
         if(Candle_Buffer[j]==6)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="SHORT_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="SHORT_Manfi";
              }
           }
         if(Candle_Buffer[j]==7)
           {
            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="STAR_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="STAR_Manfi";
              }
           }
         if(Candle_Buffer[j]==8)
           {

            if(mrate[j].open<mrate[j].close)
              {
               SymbolCandel[i][j]="MARIBOZU_ADDI_Mosbat";
              }
            else
              {
               SymbolCandel[i][j]="MARIBOZU_ADDI_Manfi";
              }
           }
        }
        


      //---------------------------------------------------------------------------------------------
      // Print("Before zigzag to sup and res_ZSTR ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      //------------------------------------------ zigzag to sup and res_2 ----------------------------------//
      if(last_price==0 && i<symbol_size)
        {
         last_price=this_day_CP[i];
        }

      if(i>=symbol_size && last_price==0)
        {
         last_price=mrate[0].close;
        }



      Strong_Signal_From_zigzag_Zstr[i]=false;

      if(ZigZag_Buffer_Zstr[0]==0 && ZigZag_Buffer_Zstr[1]!=0)// ehteam ghavi shekast ravand
        {
         Strong_Signal_From_zigzag_Zstr[i]=true;
        }

      n_bar_Change_ravand_Zstr[i]=0;            //??
      int n=0;
      if(ZigZag_Buffer_Zstr[0]==0)
        {
         Signal_From_zigzag_Zstr[i]=false;      //??
         for(int k=0;k<copied;k++)
           {

            if(ZigZag_Buffer_Zstr[k]!=0)
              {
               ArrayResize(ZigZag_Buffer_F_Zstr,n+1,10);     //??
               ZigZag_Buffer_F_Zstr[n][0]=ZigZag_Buffer_Zstr[k];  //ghemati ke taghire ravand rokh dade
               ZigZag_Buffer_F_Zstr[n][1]=k;                 // shomare candeli ke taghir ravand rokh dade(tamame ravand ha)
               if(n==0)
                 {
                  n_bar_Change_ravand_Zstr[i]=k;         // shomare candeli ke akharin taghire ravand rokh dade
                 }
               n++;
              }

           }
        }
      else
        {
         Signal_From_zigzag_Zstr[i]=true; // dar in halat ek ravand mirim aghab chon hanoz ravand stable nashode
         for(int k=1;k<copied;k++)
           {

            if(ZigZag_Buffer_Zstr[k]!=0)
              {
               ArrayResize(ZigZag_Buffer_F_Zstr,n+1,10);
               ZigZag_Buffer_F_Zstr[n][0]=ZigZag_Buffer_Zstr[k];
               ZigZag_Buffer_F_Zstr[n][1]=k;
               
               if(n==0)
                 {
                  n_bar_Change_ravand_Zstr[i]=k;
                 }
               n++;
              }

           }
        }
        
            double Sum_shib_rav_mosbat_Zst_Zst=0;
            double Sum_shib_rav_manfi_Zst=0;
            double Ave_shib_rav_mosbat_Zst=0;
            double Ave_shib_rav_manfi_Zst=0;
            double sum_UptoDo_mosb_Zst=0;
            double Ave__UptoDo_mosb_Zst=0;
            double sum_UptoDo_manf_Zst=0;
            double Ave__UptoDo_manf_Zst=0;
            int c=0;
            int d=0;
            int count_Zig_buff_Zst=ArrayRange(ZigZag_Buffer_F_Zstr,0);
            
      if(count_Zig_buff_Zst>=4)
        {
         ArrayResize(shib_ravand_Zstr,count_Zig_buff_Zst,10);
         
        for(int m=1;m<(count_Zig_buff_Zst);m++) // shib taghire darsad har ravand
          {
           shib_ravand_Zstr[m]=(((ZigZag_Buffer_F_Zstr[m-1][0]-ZigZag_Buffer_F_Zstr[m][0])/ZigZag_Buffer_F_Zstr[m-1][0])*100) // shibravand 20 haye ghabli
           /(ZigZag_Buffer_F_Zstr[m][1]-ZigZag_Buffer_F_Zstr[m-1][1]);
           
           shib_ravand_Zstr[0]=(((last_price-ZigZag_Buffer_F_Zstr[0][0])/last_price)*100)/ZigZag_Buffer_F_Zstr[0][1]; // shib ravand alan
          }
          if(count_Zig_buff_Zst>20)
            {

             for(int n=1;n<19;n++)
               {
                if(shib_ravand_Zstr[n]>0)
                  {
                   c++;
                   Sum_shib_rav_mosbat_Zst_Zst=Sum_shib_rav_mosbat_Zst_Zst+shib_ravand_Zstr[n];
                   Ave_shib_rav_mosbat_Zst=Sum_shib_rav_mosbat_Zst_Zst/c;
                   sum_UptoDo_mosb_Zst=sum_UptoDo_mosb_Zst+(((ZigZag_Buffer_F_Zstr[n-1][0]-ZigZag_Buffer_F_Zstr[n][0])/ZigZag_Buffer_F_Zstr[n-1][0])*100);
                   Ave__UptoDo_mosb_Zst=sum_UptoDo_mosb_Zst/c;
                   
                  }
                if(shib_ravand_Zstr[n]<0)
                  {
                   d++;
                   Sum_shib_rav_manfi_Zst=Sum_shib_rav_manfi_Zst+shib_ravand_Zstr[n];
                   Ave_shib_rav_manfi_Zst=Sum_shib_rav_manfi_Zst/d;
                   sum_UptoDo_manf_Zst=sum_UptoDo_manf_Zst+(((ZigZag_Buffer_F_Zstr[n-1][0]-ZigZag_Buffer_F_Zstr[n][0])/ZigZag_Buffer_F_Zstr[n-1][0])*100);
                   Ave__UptoDo_manf_Zst=sum_UptoDo_manf_Zst/d;
                  }
               }
            }
            else
              {
             for(int n=1;n<(count_Zig_buff_Zst-1);n++)
               {
                if(shib_ravand_Zstr[n]>0)
                  {
                   c++;
                   Sum_shib_rav_mosbat_Zst_Zst=Sum_shib_rav_mosbat_Zst_Zst+shib_ravand_Zstr[n];
                   Ave_shib_rav_mosbat_Zst=Sum_shib_rav_mosbat_Zst_Zst/c;
                   sum_UptoDo_mosb_Zst=sum_UptoDo_mosb_Zst+(((ZigZag_Buffer_F_Zstr[n-1][0]-ZigZag_Buffer_F_Zstr[n+1][0])/ZigZag_Buffer_F_Zstr[n-1][0])*100);
                   Ave__UptoDo_mosb_Zst=sum_UptoDo_mosb_Zst/c;
                  }
                if(shib_ravand_Zstr[n]<0)
                  {
                   d++;
                   Sum_shib_rav_manfi_Zst=Sum_shib_rav_manfi_Zst+shib_ravand_Zstr[n];
                   Ave_shib_rav_manfi_Zst=Sum_shib_rav_manfi_Zst/d;
                   sum_UptoDo_manf_Zst=sum_UptoDo_manf_Zst+(((ZigZag_Buffer_F_Zstr[n-1][0]-ZigZag_Buffer_F_Zstr[n][0])/ZigZag_Buffer_F_Zstr[n-1][0])*100);
                   Ave__UptoDo_manf_Zst=sum_UptoDo_manf_Zst/d;
                  }
               }
              }
          
          ArrayPrint(ZigZag_Buffer_F_Zstr);
        
         Chg_RavandInPrice1nd_Zstr=ZigZag_Buffer_F_Zstr[0][0];         //??
         Chg_RavandBarNumb1nd_Zstr=ZigZag_Buffer_F_Zstr[0][1];         //??

         Chg_RavandInPrice2nd_Zstr=ZigZag_Buffer_F_Zstr[1][0];         //??
         Chg_RavandBarNumb2nd_Zstr=ZigZag_Buffer_F_Zstr[1][1];         //??

         Chg_RavandInPrice3nd_Zstr=ZigZag_Buffer_F_Zstr[2][0];         //??
         Chg_RavandBarNumb3nd_Zstr=ZigZag_Buffer_F_Zstr[2][1];         //??

         Chg_RavandInPrice4nd_Zstr=ZigZag_Buffer_F_Zstr[3][0];         //??
         Chg_RavandBarNumb4nd_Zstr=ZigZag_Buffer_F_Zstr[3][1];         //??


         shib_ravand_Now_Zstr=(((last_price-Chg_RavandInPrice1nd_Zstr)/last_price)*100)/Chg_RavandBarNumb1nd_Zstr;                 //??
         shib_ravand1_Zstr=(((Chg_RavandInPrice1nd_Zstr-Chg_RavandInPrice2nd_Zstr)/Chg_RavandInPrice1nd_Zstr)*100)/(Chg_RavandBarNumb2nd_Zstr-Chg_RavandBarNumb1nd_Zstr);          //??
         shib_ravand2_Zstr=(((Chg_RavandInPrice2nd_Zstr-Chg_RavandInPrice3nd_Zstr)/Chg_RavandInPrice2nd_Zstr)*100)/(Chg_RavandBarNumb3nd_Zstr-Chg_RavandBarNumb2nd_Zstr);         //??
         shib_ravand3_Zstr=(((Chg_RavandInPrice3nd_Zstr-Chg_RavandInPrice4nd_Zstr)/Chg_RavandInPrice3nd_Zstr)*100)/(Chg_RavandBarNumb4nd_Zstr-Chg_RavandBarNumb3nd_Zstr);         //??
         
        }
      else
        {
         continue;
        }


      //------------------------------------------------------
      // Print("Before taein Ravand soodi ya nozooli ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      //-----------------------------------------taein Ravand soodi ya nozooli
double chang_ravand_INDX_Zst=Ave__UptoDo_mosb_Zst/6;




      if(Strong_Signal_From_zigzag_Zstr[i]==false)
        {

         if(shib_ravand_Zstr[0]>0)
           {
            Ravand_Zstr="Ravand Soodi";
           }
         else if(shib_ravand_Zstr[0]<0)
           {
            Ravand_Zstr="Ravand Nozooli";
           }
         else
           {
            Ravand_Zstr="Ravand mobham_Shayad_Kanal";
           }
         //Print(m_symbols_array[i],"  Time  ",Ctime,"  Ravand  ",Ravand,"ZigZag_Buffer_F[0]",ZigZag_Buffer_F[0],"if1","last_price",last_price);
         counter_Chng_Percent_Zstr[i]=0;
         LastPercent_Zstr[i]=0;
        }
      else
        {
         if(last_price>ZigZag_Buffer_F_Zstr[1][0] && (((last_price-ZigZag_Buffer_F_Zstr[1][0])/last_price)*100)>=chang_ravand_INDX_Zst)
           {
            Ravand_Zstr="Ravand_Transient Soodi To Nozooli";
           }
         else if(last_price<ZigZag_Buffer_F_Zstr[1][0] && (((last_price-ZigZag_Buffer_F_Zstr[1][0])/last_price)*100)<=-1*chang_ravand_INDX_Zst)
           {
            Ravand_Zstr="Ravand_Transient Nozooli To Soodi";
           }
         else
           {
            Ravand_Zstr="Ravand_Transient_Ravand_Mobham";
           }
         //Print(m_symbols_array[i],"  Time  ",Ctime,"  Ravand  ",Ravand,"ZigZag_Buffer_F[1]",ZigZag_Buffer_F[1],"if3","last_price",last_price);
        }
        
        if(ZigZag_Buffer_Zstr[0]==0 && ZigZag_Buffer_Zstr[1]==0)
          {
           Ravand_Zstr="Ravand mobham_Shayad_Kanal";
          }
          



      Ravand_symbol_Zstr[i]=Ravand_Zstr;

      // ------date of change Ravand----

      string date_of_Ch_Ravand_Zstr=TimeToString(mrate[n_bar_Change_ravand_Zstr[i]].time,TIME_DATE);
      //Print(m_symbols_array[i],"  date_of_Ch_Ravand  ",date_of_Ch_Ravand,"  n_bar_Change_ravand[i]  ",n_bar_Change_ravand[i]);
      //---------------------------
      //Filter 2// hadaf= sup and ress hae nazdik ra yeki mikonad + sort az kam be ziad mikonad
      n=0;
      ArraySort(ZigZag_Buffer_F_Zstr);
      int ZigZag_Buffer_F_size_Zstr=ArrayRange(ZigZag_Buffer_F_Zstr,0);
      ArrayResize(ZigZag_Buffer_F2_Zstr,n+1,10);                                //??
      ZigZag_Buffer_F2_Zstr[n][0]=(ZigZag_Buffer_F_Zstr[0][0]);
      ZigZag_Buffer_F2_Zstr[n][1]=1;


         for(int a=1;a<(ZigZag_Buffer_F_size_Zstr);a++)
           {
            if(((ZigZag_Buffer_F_Zstr[a][0]-ZigZag_Buffer_F2_Zstr[n][0])
            /ZigZag_Buffer_F2_Zstr[n][0])*100<chang_ravand_INDX_Zst) // saghf va kaf haye kamtar az 7 darsad ekhtelaf, yeki mishan
              {
               ZigZag_Buffer_F2_Zstr[n][1]=ZigZag_Buffer_F2_Zstr[n][1]+1; //

               ZigZag_Buffer_F2_Zstr[n][0]=(ZigZag_Buffer_F_Zstr[a][0]        // amalan ye miangin migire az kol sup & ress nazdik
               +((ZigZag_Buffer_F2_Zstr[n][1]-1)*ZigZag_Buffer_F2_Zstr[n][0]))/(ZigZag_Buffer_F2_Zstr[n][1]);

              }
            else
              {
               n++;
               ArrayResize(ZigZag_Buffer_F2_Zstr,n+1,10);
               ZigZag_Buffer_F2_Zstr[n][1]=1;
               ZigZag_Buffer_F2_Zstr[n][0]=ZigZag_Buffer_F_Zstr[a][0];
               //ZigZag_Buffer_F2[n][1]=ZigZag_Buffer_F2[n][1]+1; //avalesh sefr shavad

              }
           }



      //Print("ZigZag_Buffer_F");
      //ArrayPrint(ZigZag_Buffer_F);
      //Print("ZigZag_Buffer_F2");
      //ArrayPrint(ZigZag_Buffer_F2);
//---------------------------------- this ravand percent and last ravand percent----------------------//
      double now_ravand_perc=0;
      double sec_ravand_perc=0;
      double third_ravand_perc=0;
if(Ravand_Zstr=="Ravand Soodi" || Ravand_Zstr=="Ravand Nozooli")
  {
    now_ravand_perc=((last_price-Chg_RavandInPrice1nd_Zstr)/Chg_RavandInPrice1nd_Zstr)*100;
    sec_ravand_perc=((Chg_RavandInPrice1nd_Zstr-Chg_RavandInPrice2nd_Zstr)/Chg_RavandInPrice2nd_Zstr)*100;
    third_ravand_perc=((Chg_RavandInPrice2nd_Zstr-Chg_RavandInPrice3nd_Zstr)/Chg_RavandInPrice3nd_Zstr)*100;
  }
  else
    {
    now_ravand_perc=0;
    sec_ravand_perc=((Chg_RavandInPrice1nd_Zstr-Chg_RavandInPrice2nd_Zstr)/Chg_RavandInPrice2nd_Zstr)*100;
    third_ravand_perc=((Chg_RavandInPrice2nd_Zstr-Chg_RavandInPrice3nd_Zstr)/Chg_RavandInPrice3nd_Zstr)*100;
    }


      //--------------------------------------------------------------------------------
      //  Print("Before tabdil saghf va kaf ha ",m_symbols_array[i],
      //  " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);    
      //----------------tabdil saghf va kaf ha be sep va ress ba tavajoh be ravand va gheymat

      Supp_Z_Zstr=0;                //??
      Supp_Z_STR_Zstr=0;                //??
      Ress_Z_Zstr=0;                //??
      Ress_Z_STR_Zstr=0;                //??
double Perc_LastToF2m_Zstr=0;
      int ZigZag_Buffer_F2_size_Zstr=ArrayRange(ZigZag_Buffer_F2_Zstr,0);

      if(Ravand_Zstr=="Ravand Soodi" || Ravand_Zstr=="Ravand_Transient Nozooli To Soodi")
        {
         for(int m=0;m<ZigZag_Buffer_F2_size_Zstr;m++)
           {

             Perc_LastToF2m_Zstr=((last_price-ZigZag_Buffer_F2_Zstr[m][0])/last_price)*100;

            if(Ravand_Zstr=="Ravand Soodi"
               && Perc_LastToF2m_Zstr<=(-2*chang_ravand_INDX_Zst) && (m-1)!=-1)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m-1][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m-1][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               break;
              }

            if(Ravand_Zstr=="Ravand Soodi"
               && Perc_LastToF2m_Zstr>=-2*chang_ravand_INDX_Zst && Perc_LastToF2m_Zstr<chang_ravand_INDX_Zst && (m-1)!=-1)// nabayad talaghi dashteh bashad chon zir 2.5 ravand avaz mishe
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m-1][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m-1][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               break;
              }

            if(m==ZigZag_Buffer_F2_size_Zstr-1 && Ravand_Zstr=="Ravand Soodi"
               && Perc_LastToF2m_Zstr>chang_ravand_INDX_Zst)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               Ress_Z_Zstr=0;
               Ress_Z_STR_Zstr=0;
               break;
              }

            if(Ravand_Zstr=="Ravand_Transient Nozooli To Soodi"
               && Perc_LastToF2m_Zstr<=-2*chang_ravand_INDX_Zst && (m-1)!=-1)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m-1][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m-1][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               break;
              }

            if(Ravand_Zstr=="Ravand_Transient Nozooli To Soodi"
               && Perc_LastToF2m_Zstr>=-1*chang_ravand_INDX_Zst && Perc_LastToF2m_Zstr<=chang_ravand_INDX_Zst && (m+1)!=ZigZag_Buffer_F2_size_Zstr)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m+1][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m+1][1];
               break;
              }

            if(m==ZigZag_Buffer_F2_size_Zstr-1 && Ravand_Zstr=="Ravand_Transient Nozooli To Soodi")
              {
               Print(m_symbols_array[i],"not define Sup and Ress_Zstr");
              }

            if(m==ZigZag_Buffer_F2_size_Zstr-1 && Ravand_Zstr=="Ravand Soodi")
              {
               Print(m_symbols_array[i],"not define Sup and Ress_Zstr");
              }

           }
        }

      if(Ravand_Zstr=="Ravand Nozooli" || Ravand_Zstr=="Ravand_Transient Soodi To Nozooli")
        {
         for(int m=ZigZag_Buffer_F2_size_Zstr-1;m>=0;m--)
           {
/*
      Print("symbol",m_symbols_array[i],"m",m);
      Print("ZigZag_Buffer_F2");
      ArrayPrint(ZigZag_Buffer_F2);
      Print("Chg_RavandInPrice1nd",Chg_RavandInPrice1nd);
      Print("ZigZag_Buffer_F2[m][0]",ZigZag_Buffer_F2[m][0]);
      */
            if(i<symbol_size)
              {
               if(((ZigZag_Buffer_F2_Zstr[m][0]-Chg_RavandInPrice1nd_Zstr)/Chg_RavandInPrice1nd_Zstr)*100<2*chang_ravand_INDX_Zst && ((ZigZag_Buffer_F2_Zstr[m][0]-Chg_RavandInPrice1nd_Zstr)/Chg_RavandInPrice1nd_Zstr)*100>-2*chang_ravand_INDX_Zst)
                 {
                  ZigZag_Buffer_F2_Zstr[m][0]=Chg_RavandInPrice1nd_Zstr;
                 }
              }
            else
              {
               if(((ZigZag_Buffer_F2_Zstr[m][0]-Chg_RavandInPrice1nd_Zstr)/Chg_RavandInPrice1nd_Zstr)*100<chang_ravand_INDX_Zst && ((ZigZag_Buffer_F2_Zstr[m][0]-Chg_RavandInPrice1nd_Zstr)/Chg_RavandInPrice1nd_Zstr)*100>-1*chang_ravand_INDX_Zst)
                 {
                  ZigZag_Buffer_F2_Zstr[m][0]=Chg_RavandInPrice1nd_Zstr;
                 }
              }

            double Perc_LastToF2m_Zstr=((last_price-ZigZag_Buffer_F2_Zstr[m][0])/last_price)*100;

            //Print("Perc_LastToF2m",Perc_LastToF2m);
            //Print("Ravand",Ravand);
            //Print("last_price",last_price);

            //Print(m_symbols_array[i],"Perc_LastToF2m",Perc_LastToF2m);
            if(Ravand_Zstr=="Ravand Nozooli"
               && Perc_LastToF2m_Zstr>=2*chang_ravand_INDX_Zst && (m+1)!=ZigZag_Buffer_F2_size_Zstr)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m+1][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m+1][1];
               break;
              }

            if(Ravand_Zstr=="Ravand Nozooli"
               && Perc_LastToF2m_Zstr<=2*chang_ravand_INDX_Zst && Perc_LastToF2m_Zstr>-1*chang_ravand_INDX_Zst && (m+1)!=ZigZag_Buffer_F2_size_Zstr)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m+1][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m+1][1];
               break;
              }

            if(m==0 && Ravand_Zstr=="Ravand Nozooli"
               && Perc_LastToF2m_Zstr<-1*chang_ravand_INDX_Zst)
              {
               Supp_Z_Zstr=0;
               Supp_Z_STR_Zstr=0;
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               break;
              }

            if(Ravand_Zstr=="Ravand_Transient Soodi To Nozooli"
               && Perc_LastToF2m_Zstr>=2*chang_ravand_INDX_Zst && (m+1)!=ZigZag_Buffer_F2_size_Zstr)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m+1][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m+1][1];
               break;
              }

            if(Ravand_Zstr=="Ravand_Transient Soodi To Nozooli"
               && Perc_LastToF2m_Zstr<=chang_ravand_INDX_Zst && Perc_LastToF2m_Zstr>=-1*chang_ravand_INDX_Zst && (m-1)!=-1)
              {
               Supp_Z_Zstr=ZigZag_Buffer_F2_Zstr[m-1][0];
               Supp_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m-1][1];
               Ress_Z_Zstr=ZigZag_Buffer_F2_Zstr[m][0];
               Ress_Z_STR_Zstr=ZigZag_Buffer_F2_Zstr[m][1];
               break;
              }

            if(m==0 && Ravand_Zstr=="Ravand Nozooli")
              {
               Print(m_symbols_array[i],"not define Sup and Ress_Zstr");
              }
            if(m==0 && Ravand_Zstr=="Ravand_Transient Soodi To Nozooli")
              {
               Print(m_symbols_array[i],"not define Sup and Ress_Zstr");
              }
           }
        }

      double Last_Souod_Or_Nozol_perc_Zstr=((last_price-Chg_RavandInPrice1nd_Zstr)/last_price)*100; //taein darsad Akharin taghir ravand ta alan

                                                                                                    //Print(m_symbols_array[i],"Ravand",Ravand," date_of_Ch_Ravand ",date_of_Ch_Ravand," n_bar_Change_ravand[i] ",n_bar_Change_ravand[i]);
      //Print(m_symbols_array[i],"Ress",Ress_Z," Ress_STR ",Ress_Z_STR," Sup ",Supp_Z,"Sup STR",Supp_Z_STR);

      Supp_Z_symbol_Zstr[i]=Supp_Z_Zstr;         //??
      Ress_Z_symbol_Zstr[i]=Ress_Z_Zstr;         //??
      Supp_Z_STR_symbol_Zstr[i]=Supp_Z_STR_Zstr;         //??
      Ress_Z_STR_symbol_Zstr[i]=Ress_Z_STR_Zstr;         //??

      //---------------------------------------------------------------------------------------------
      // Print("Before zigzag to sup and res_Avalie ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      //------------------------------------------ zigzag to sup and res ----------------------------------//
      if(last_price==0 && i<symbol_size)
        {
         last_price=this_day_CP[i];
        }

      if(i>=symbol_size && last_price==0)
        {
         last_price=mrate[0].close;
        }



      Strong_Signal_From_zigzag[i]=false;

      if(ZigZag_Buffer[0]==0 && ZigZag_Buffer[1]!=0)
        {
         Strong_Signal_From_zigzag[i]=true;
        }

      n_bar_Change_ravand[i]=0;
      n=0;
      if(ZigZag_Buffer[0]==0)
        {
         Signal_From_zigzag[i]=false;
         for(int k=0;k<copied;k++)
           {

            if(ZigZag_Buffer[k]!=0)
              {
               ArrayResize(ZigZag_Buffer_F,n+1,10);
               ZigZag_Buffer_F[n][0]=ZigZag_Buffer[k];
               ZigZag_Buffer_F[n][1]=k;
               if(n==0)
                 {
                  n_bar_Change_ravand[i]=k;
                 }
               n++;
              }

           }
        }
      else
        {
         Signal_From_zigzag[i]=true;
         for(int k=1;k<copied;k++)
           {

            if(ZigZag_Buffer[k]!=0)
              {
               ArrayResize(ZigZag_Buffer_F,n+1,10);
               ZigZag_Buffer_F[n][0]=ZigZag_Buffer[k];
               ZigZag_Buffer_F[n][1]=k;
               if(n==0)
                 {
                  n_bar_Change_ravand[i]=k;
                 }
               n++;
              }

           }
        }


            double Sum_shib_rav_mosbat=0;
            double Sum_shib_rav_manfi=0;
            double Ave_shib_rav_mosbat=0;
            double Ave_shib_rav_manfi=0;
            double sum_UptoDo_mosb=0;
            double Ave__UptoDo_mosb=0;
            double sum_UptoDo_manf=0;
            double Ave__UptoDo_manf=0;
             c=0;
             d=0;
            int count_Zig_buff=ArrayRange(ZigZag_Buffer_F,0);
        
        
      if(ArrayRange(ZigZag_Buffer_F,0)>=4)
        {
        
         ArrayResize(shib_ravand,count_Zig_buff,10);
         
        for(int m=1;m<(count_Zig_buff);m++) // shib taghire darsad har ravand
          {
           shib_ravand[m]=(((ZigZag_Buffer_F[m-1][0]-ZigZag_Buffer_F[m][0])/ZigZag_Buffer_F[m-1][0])*100) // shibravand 20 haye ghabli
           /(ZigZag_Buffer_F[m][1]-ZigZag_Buffer_F[m-1][1]);
           
           shib_ravand[0]=(((last_price-ZigZag_Buffer_F[0][0])/last_price)*100)/ZigZag_Buffer_F[0][1]; // shib ravand alan
          }
          if(count_Zig_buff>20)
            {

             for(int n=1;n<19;n++)
               {
                if(shib_ravand[n]>0)
                  {
                   c++;
                   Sum_shib_rav_mosbat=Sum_shib_rav_mosbat+shib_ravand[n];
                   Ave_shib_rav_mosbat=Sum_shib_rav_mosbat/c;
                   sum_UptoDo_mosb=sum_UptoDo_mosb+(((ZigZag_Buffer_F[n-1][0]-ZigZag_Buffer_F[n][0])/ZigZag_Buffer_F[n-1][0])*100);
                   Ave__UptoDo_mosb=sum_UptoDo_mosb/c;
                   
                  }
                if(shib_ravand[n]<0)
                  {
                   d++;
                   Sum_shib_rav_manfi=Sum_shib_rav_manfi+shib_ravand[n];
                   Ave_shib_rav_manfi=Sum_shib_rav_manfi/d;
                   sum_UptoDo_manf=sum_UptoDo_manf+(((ZigZag_Buffer_F[n-1][0]-ZigZag_Buffer_F[n][0])/ZigZag_Buffer_F[n-1][0])*100);
                   Ave__UptoDo_manf=sum_UptoDo_manf/d;
                  }
               }
            }
            else
              {
             for(int n=1;n<(count_Zig_buff-1);n++)
               {
                if(shib_ravand[n]>0)
                  {
                   c++;
                   Sum_shib_rav_mosbat=Sum_shib_rav_mosbat+shib_ravand[n];
                   Ave_shib_rav_mosbat=Sum_shib_rav_mosbat/c;
                   sum_UptoDo_mosb=sum_UptoDo_mosb+(((ZigZag_Buffer_F[n-1][0]-ZigZag_Buffer_F[n+1][0])/ZigZag_Buffer_F[n-1][0])*100);
                   Ave__UptoDo_mosb=sum_UptoDo_mosb/c;
                  }
                if(shib_ravand[n]<0)
                  {
                   d++;
                   Sum_shib_rav_manfi=Sum_shib_rav_manfi+shib_ravand[n];
                   Ave_shib_rav_manfi=Sum_shib_rav_manfi/d;
                   sum_UptoDo_manf=sum_UptoDo_manf+(((ZigZag_Buffer_F[n-1][0]-ZigZag_Buffer_F[n][0])/ZigZag_Buffer_F[n-1][0])*100);
                   Ave__UptoDo_manf=sum_UptoDo_manf/d;
                  }
               }
              }
          
   
        
        
         Chg_RavandInPrice1nd=ZigZag_Buffer_F[0][0];
         Chg_RavandBarNumb1nd=ZigZag_Buffer_F[0][1];

         Chg_RavandInPrice2nd=ZigZag_Buffer_F[1][0];
         Chg_RavandBarNumb2nd=ZigZag_Buffer_F[1][1];

         Chg_RavandInPrice3nd=ZigZag_Buffer_F[2][0];
         Chg_RavandBarNumb3nd=ZigZag_Buffer_F[2][1];

         Chg_RavandInPrice4nd=ZigZag_Buffer_F[3][0];
         Chg_RavandBarNumb4nd=ZigZag_Buffer_F[3][1];


         shib_ravand_Now=(((last_price-Chg_RavandInPrice1nd)/last_price)*100)/Chg_RavandBarNumb1nd;
         shib_ravand1=(((Chg_RavandInPrice1nd-Chg_RavandInPrice2nd)/Chg_RavandInPrice1nd)*100)/(Chg_RavandBarNumb2nd-Chg_RavandBarNumb1nd);
         shib_ravand2=(((Chg_RavandInPrice2nd-Chg_RavandInPrice3nd)/Chg_RavandInPrice2nd)*100)/(Chg_RavandBarNumb3nd-Chg_RavandBarNumb2nd);
         shib_ravand3=(((Chg_RavandInPrice3nd-Chg_RavandInPrice4nd)/Chg_RavandInPrice3nd)*100)/(Chg_RavandBarNumb4nd-Chg_RavandBarNumb3nd);

        }
      else
        {
         continue;
        }


      //------------------------------------------------------
      // Print("Before taein Ravand soodi ya nozooli ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);
      //-----------------------------------------taein Ravand soodi ya nozooli
double chang_ravand_INDX=Ave__UptoDo_mosb/6;

      if(Strong_Signal_From_zigzag[i]==false)
        {

         if(shib_ravand[0]>0)
           {
            Ravand="Ravand Soodi";
           }
         else if(shib_ravand[0]<0)
           {
            Ravand="Ravand Nozooli";
           }
         else
           {
            Ravand="Ravand mobham_Shayad_Kanal";
           }
         //Print(m_symbols_array[i],"  Time  ",Ctime,"  Ravand  ",Ravand,"ZigZag_Buffer_F[0]",ZigZag_Buffer_F[0],"if1","last_price",last_price);
         counter_Chng_Percent[i]=0;
         LastPercent[i]=0;
        }
      else
        {
         if(last_price>ZigZag_Buffer_F[1][0] && (((last_price-ZigZag_Buffer_F[1][0])/last_price)*100)>=chang_ravand_INDX)
           {
            Ravand="Ravand_Transient Soodi To Nozooli";
           }
         else if(last_price<ZigZag_Buffer_F[1][0] && (((last_price-ZigZag_Buffer_F[1][0])/last_price)*100)<=-1*chang_ravand_INDX)
           {
            Ravand="Ravand_Transient Nozooli To Soodi";
           }
         else
           {
            Ravand="Ravand_Transient_Ravand_Mobham";
           }
         //Print(m_symbols_array[i],"  Time  ",Ctime,"  Ravand  ",Ravand,"ZigZag_Buffer_F[1]",ZigZag_Buffer_F[1],"if3","last_price",last_price);
        }
        
        if(ZigZag_Buffer[0]==0 && ZigZag_Buffer[1]==0)
          {
           Ravand="Ravand mobham_Shayad_Kanal";
          }

          
      Ravand_symbol[i]=Ravand;

      // ------date of change Ravand----

      string date_of_Ch_Ravand=TimeToString(mrate[n_bar_Change_ravand[i]].time,TIME_DATE);
      //Print(m_symbols_array[i],"  date_of_Ch_Ravand  ",date_of_Ch_Ravand,"  n_bar_Change_ravand[i]  ",n_bar_Change_ravand[i]);
      //---------------------------
      //Filter 2
      n=0;
      ArraySort(ZigZag_Buffer_F);
      int ZigZag_Buffer_F_size=ArrayRange(ZigZag_Buffer_F,0);
      ArrayResize(ZigZag_Buffer_F2,n+1,10);
      ZigZag_Buffer_F2[n][0]=(ZigZag_Buffer_F[0][0]);
      ZigZag_Buffer_F2[n][1]=1;

      if(i<symbol_size)
        {
         for(int a=1;a<(ZigZag_Buffer_F_size);a++)
           {
            if(((ZigZag_Buffer_F[a][0]-ZigZag_Buffer_F2[n][0])/ZigZag_Buffer_F2[n][0])*100<2*chang_ravand_INDX) // saghf va kaf haye kamtar az 7 darsad ekhtelaf, yeki mishan
              {
               ZigZag_Buffer_F2[n][1]=ZigZag_Buffer_F2[n][1]+1; //avalesh sefr shavad

               ZigZag_Buffer_F2[n][0]=(ZigZag_Buffer_F[a][0]+((ZigZag_Buffer_F2[n][1]-1)*ZigZag_Buffer_F2[n][0]))/(ZigZag_Buffer_F2[n][1]);

              }
            else
              {
               n++;
               ArrayResize(ZigZag_Buffer_F2,n+1,10);
               ZigZag_Buffer_F2[n][1]=1;
               ZigZag_Buffer_F2[n][0]=ZigZag_Buffer_F[a][0];
               //ZigZag_Buffer_F2[n][1]=ZigZag_Buffer_F2[n][1]+1; //avalesh sefr shavad

              }
           }
        }
      else
        {
         for(int a=1;a<(ZigZag_Buffer_F_size);a++)
           {
            if(((ZigZag_Buffer_F[a][0]-ZigZag_Buffer_F2[n][0])/ZigZag_Buffer_F2[n][0])*100<=chang_ravand_INDX) // saghf va kaf haye kamtar az 7 darsad ekhtelaf, yeki mishan
              {
               ZigZag_Buffer_F2[n][1]=ZigZag_Buffer_F2[n][1]+1; //avalesh sefr shavad

               ZigZag_Buffer_F2[n][0]=(ZigZag_Buffer_F[a][0]+((ZigZag_Buffer_F2[n][1]-1)*ZigZag_Buffer_F2[n][0]))/(ZigZag_Buffer_F2[n][1]);

              }
            else
              {
               n++;
               ArrayResize(ZigZag_Buffer_F2,n+1,10);
               ZigZag_Buffer_F2[n][1]=1;
               ZigZag_Buffer_F2[n][0]=ZigZag_Buffer_F[a][0];
               //ZigZag_Buffer_F2[n][1]=ZigZag_Buffer_F2[n][1]+1; //avalesh sefr shavad

              }
           }
        }

      //Print("ZigZag_Buffer_F");
      //ArrayPrint(ZigZag_Buffer_F);
      //Print("ZigZag_Buffer_F2");
      //ArrayPrint(ZigZag_Buffer_F2);


      //--------------------------------------------------------------------------------
      //  Print("Before tabdil saghf va kaf ha ",m_symbols_array[i],
      //  " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);    
      //----------------tabdil saghf va kaf ha be sep va ress ba tavajoh be ravand va gheymat

      Supp_Z=0;
      Supp_Z_STR=0;
      Ress_Z=0;
      Ress_Z_STR=0;

      int ZigZag_Buffer_F2_size=ArrayRange(ZigZag_Buffer_F2,0);

      if(Ravand=="Ravand Soodi" || Ravand=="Ravand_Transient Nozooli To Soodi")
        {
         for(int m=0;m<ZigZag_Buffer_F2_size;m++)
           {
/*
      Print("symbol",m_symbols_array[i],"m",m);
      Print("ZigZag_Buffer_F2");
      ArrayPrint(ZigZag_Buffer_F2);
      Print("Chg_RavandInPrice1nd",Chg_RavandInPrice1nd);
      Print("ZigZag_Buffer_F2[m][0]",ZigZag_Buffer_F2[m][0]);
      */
            if(i<symbol_size)
              {
               if(((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100<2*chang_ravand_INDX
                  && ((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100>-2*chang_ravand_INDX)
                 {
                  ZigZag_Buffer_F2[m][0]=Chg_RavandInPrice1nd;
                 }
              }
            else
              {
               if(((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100<chang_ravand_INDX
                  && ((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100>-1*chang_ravand_INDX)
                 {
                  ZigZag_Buffer_F2[m][0]=Chg_RavandInPrice1nd;
                 }
              }
            double Perc_LastToF2m=((last_price-ZigZag_Buffer_F2[m][0])/last_price)*100;

            //Print("Perc_LastToF2m",Perc_LastToF2m);
            //Print("Ravand",Ravand);
            //Print("last_price",last_price);

            //Print(m_symbols_array[i],"Perc_LastToF2m",Perc_LastToF2m);
            if(Ravand=="Ravand Soodi"
               && Perc_LastToF2m<=-2*chang_ravand_INDX && (m-1)!=-1)
              {
               Supp_Z=ZigZag_Buffer_F2[m-1][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m-1][1];
               Ress_Z=ZigZag_Buffer_F2[m][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m][1];
               break;
              }

            if(Ravand=="Ravand Soodi"
               && Perc_LastToF2m>=-2*chang_ravand_INDX && Perc_LastToF2m<chang_ravand_INDX && (m-1)!=-1)// nabayad talaghi dashteh bashad chon zir 2.5 ravand avaz mishe
              {
               Supp_Z=ZigZag_Buffer_F2[m-1][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m-1][1];
               Ress_Z=ZigZag_Buffer_F2[m][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m][1];
               break;
              }

            if(m==ZigZag_Buffer_F2_size-1 && Ravand=="Ravand Soodi"
               && Perc_LastToF2m>chang_ravand_INDX)
              {
               Supp_Z=ZigZag_Buffer_F2[m][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m][1];
               Ress_Z=0;
               Ress_Z_STR=0;
               break;
              }

            if(Ravand=="Ravand_Transient Nozooli To Soodi"
               && Perc_LastToF2m<=-2*chang_ravand_INDX && (m-1)!=-1)
              {
               Supp_Z=ZigZag_Buffer_F2[m-1][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m-1][1];
               Ress_Z=ZigZag_Buffer_F2[m][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m][1];
               break;
              }

            if(Ravand=="Ravand_Transient Nozooli To Soodi"
               && Perc_LastToF2m>=-1*chang_ravand_INDX && Perc_LastToF2m<=chang_ravand_INDX && (m+1)!=ZigZag_Buffer_F2_size)
              {
               Supp_Z=ZigZag_Buffer_F2[m][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m][1];
               Ress_Z=ZigZag_Buffer_F2[m+1][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m+1][1];
               break;
              }

            if(m==ZigZag_Buffer_F2_size-1 && Ravand=="Ravand_Transient Nozooli To Soodi")
              {
               Print(m_symbols_array[i],"not define Sup and Ress");
              }

            if(m==ZigZag_Buffer_F2_size-1 && Ravand=="Ravand Soodi")
              {
               Print(m_symbols_array[i],"not define Sup and Ress");
              }

           }
        }

      if(Ravand=="Ravand Nozooli" || Ravand=="Ravand_Transient Soodi To Nozooli")
        {
         for(int m=ZigZag_Buffer_F2_size-1;m>=0;m--)
           {
/*
      Print("symbol",m_symbols_array[i],"m",m);
      Print("ZigZag_Buffer_F2");
      ArrayPrint(ZigZag_Buffer_F2);
      Print("Chg_RavandInPrice1nd",Chg_RavandInPrice1nd);
      Print("ZigZag_Buffer_F2[m][0]",ZigZag_Buffer_F2[m][0]);
      */
            if(i<symbol_size)
              {
               if(((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100<2*chang_ravand_INDX && ((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100>-2*chang_ravand_INDX)
                 {
                  ZigZag_Buffer_F2[m][0]=Chg_RavandInPrice1nd;
                 }
              }
            else
              {
               if(((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100<chang_ravand_INDX && ((ZigZag_Buffer_F2[m][0]-Chg_RavandInPrice1nd)/Chg_RavandInPrice1nd)*100>-1*chang_ravand_INDX)
                 {
                  ZigZag_Buffer_F2[m][0]=Chg_RavandInPrice1nd;
                 }
              }

            double Perc_LastToF2m=((last_price-ZigZag_Buffer_F2[m][0])/last_price)*100;

            //Print("Perc_LastToF2m",Perc_LastToF2m);
            //Print("Ravand",Ravand);
            //Print("last_price",last_price);

            //Print(m_symbols_array[i],"Perc_LastToF2m",Perc_LastToF2m);
            if(Ravand=="Ravand Nozooli"
               && Perc_LastToF2m>=2*chang_ravand_INDX && (m+1)!=ZigZag_Buffer_F2_size)
              {
               Supp_Z=ZigZag_Buffer_F2[m][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m][1];
               Ress_Z=ZigZag_Buffer_F2[m+1][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m+1][1];
               break;
              }

            if(Ravand=="Ravand Nozooli"
               && Perc_LastToF2m<=2*chang_ravand_INDX && Perc_LastToF2m>-1*chang_ravand_INDX && (m+1)!=ZigZag_Buffer_F2_size)
              {
               Supp_Z=ZigZag_Buffer_F2[m][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m][1];
               Ress_Z=ZigZag_Buffer_F2[m+1][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m+1][1];
               break;
              }

            if(m==0 && Ravand=="Ravand Nozooli"
               && Perc_LastToF2m<-1*chang_ravand_INDX)
              {
               Supp_Z=0;
               Supp_Z_STR=0;
               Ress_Z=ZigZag_Buffer_F2[m][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m][1];
               break;
              }

            if(Ravand=="Ravand_Transient Soodi To Nozooli"
               && Perc_LastToF2m>=2*chang_ravand_INDX && (m+1)!=ZigZag_Buffer_F2_size)
              {
               Supp_Z=ZigZag_Buffer_F2[m][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m][1];
               Ress_Z=ZigZag_Buffer_F2[m+1][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m+1][1];
               break;
              }

            if(Ravand=="Ravand_Transient Soodi To Nozooli"
               && Perc_LastToF2m<=chang_ravand_INDX && Perc_LastToF2m>=-1*chang_ravand_INDX && (m-1)!=-1)
              {
               Supp_Z=ZigZag_Buffer_F2[m-1][0];
               Supp_Z_STR=ZigZag_Buffer_F2[m-1][1];
               Ress_Z=ZigZag_Buffer_F2[m][0];
               Ress_Z_STR=ZigZag_Buffer_F2[m][1];
               break;
              }

            if(m==0 && Ravand=="Ravand Nozooli")
              {
               Print(m_symbols_array[i],"not define Sup and Ress");
              }
            if(m==0 && Ravand=="Ravand_Transient Soodi To Nozooli")
              {
               Print(m_symbols_array[i],"not define Sup and Ress");
              }
           }
        }

      double Last_Souod_Or_Nozol_perc=((last_price-Chg_RavandInPrice1nd)/last_price)*100; //taein darsad Akharin taghir ravand ta alan

                                                                                          //Print(m_symbols_array[i],"Ravand",Ravand," date_of_Ch_Ravand ",date_of_Ch_Ravand," n_bar_Change_ravand[i] ",n_bar_Change_ravand[i]);
      //Print(m_symbols_array[i],"Ress",Ress_Z," Ress_STR ",Ress_Z_STR," Sup ",Supp_Z,"Sup STR",Supp_Z_STR);

      Supp_Z_symbol[i]=Supp_Z;
      Ress_Z_symbol[i]=Ress_Z;
      Supp_Z_STR_symbol[i]=Supp_Z_STR;
      Ress_Z_STR_symbol[i]=Ress_Z_STR;
      //--------------------------------------------------
      //Print("Before Deapth Market Check And Analysis  ",m_symbols_array[i],
      //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);    
      ///-----------------------------------------------
      ///----------------------sepration of various timeframe results ------------
      ///-----------------------------------------------
Supp_Z_TF[t]=Supp_Z;
Ress_Z_TF[t]=Ress_Z;
Supp_Z_STR_TF[t]=Supp_Z_STR;
Ress_Z_STR_TF[t]=Ress_Z_STR;
Ravand_TF[t]=Ravand ;
Ravand_Zstr_TF[t]=Ravand_Zstr;

n_bar_Change_ravand_TF[t]=n_bar_Change_ravand[i];
n_bar_Change_ravand_Zstr_TF[t]=n_bar_Change_ravand_Zstr[i];


shib_ravand_Now_TF[t]=shib_ravand_Now;
shib_ravand1_TF[t]=shib_ravand1;
shib_ravand2_TF[t]=shib_ravand2;
shib_ravand3_TF[t]=shib_ravand3;
shib_ravand_Now_Zstr_TF[t]=shib_ravand_Now_Zstr;
shib_ravand1_Zstr_TF[t]=shib_ravand1_Zstr;
shib_ravand2_Zstr_TF[t]=shib_ravand2_Zstr;
shib_ravand3_Zstr_TF[t]=shib_ravand3_Zstr;

now_ravand_perc_TF[t]=now_ravand_perc;
sec_ravand_perc_TF[t]=sec_ravand_perc;
third_ravand_perc_TF[t]=third_ravand_perc;

for(int h=0;h<=9;h++)
  {
stochBuff_OSC_TF[h][t]=stochBuff_OSC[h];
signalBuff_OSC_TF[h][t]=signalBuff_OSC[h];
RSI_TF[h][t]=RSI[h];
SAR_TF[h][t]=SAR[h];
MACDBuffer_TF[h][t]=MACDBuffer[h];
MACDSignal_TF[h][t]=MACDSignal[h];
  }

//indicators

deltaMACD1_TF[t]=deltaMACD1;
deltaMACD2_TF[t]=deltaMACD2;
deltaMACD3_TF[t]=deltaMACD3;
deltaMACD_Sig1_TF[t]=deltaMACD_Sig1;
deltaMACD_Sig2_TF[t]=deltaMACD_Sig2;
deltaMACD_Sig3_TF[t]=deltaMACD_Sig3;
shibMACD1_TF[t]=shibMACD1;
shibMACD2_TF[t]=shibMACD2;
LastN_MACD_chng_TF[t]=LastN_MACD_chng;

	  // -----------------------------------------------------INDICATORS ANALYSIS Status
	  
	  //-----------RSI ANALYSIS
	  

   double rsi_now = RSI_TF[0][t];
   double rsi_prev = RSI_TF[1][t];
	  
   if(rsi_now > 70 && rsi_prev <= 70)
      RSI_Status[t] = "Vorood_OB_Ehtiait_Nozool";
   else if(rsi_now < 30 && rsi_prev >= 30)
      RSI_Status[t] =  "Vorood_OS_Ehtemal_Souod";
   else if(rsi_now < 70 && rsi_prev >= 70)
      RSI_Status[t] =  "Exit_OB_Signal_Nozool";
   else if(rsi_now > 30 && rsi_prev <= 30)
      RSI_Status[t] =  "Exit_OS_Signal_Souod";
   else if(rsi_now > 70)
      RSI_Status[t] =  "OB_Ehtiait_Nozool";
   else if(rsi_now < 30)
      RSI_Status[t] =  "OS_Ehtiait_Souod";
   else
      RSI_Status[t] =  "Neutral";
	  
	  //-----------MACD ANALYSIS
	  

   if(MACDBuffer_TF[1][t] < MACDSignal_TF[1][t] && MACDBuffer_TF[0][t] > MACDSignal_TF[0][t])
       MACD_Status[t] = "Ehtemale_Start_Souod";
   else if(MACDBuffer_TF[1][t] > MACDSignal_TF[1][t] && MACDBuffer_TF[0][t] < MACDSignal_TF[0][t])
      MACD_Status[t] = "Ehtemale_Start_Nozool";
   else if(MACDBuffer_TF[0][t] > MACDSignal_TF[0][t])
      MACD_Status[t] =  "Edame_Ravand_Souod";
   else if(MACDBuffer_TF[0][t] < MACDSignal_TF[0][t])
      MACD_Status[t] =  "Edame_Ravand_Nozool";
   else
      MACD_Status[t] =  "No_Change";


  
	  //-----------  Stochastic ANALYSIS
	  

   double k_now = stochBuff_OSC_TF[0][t];
   double d_now = signalBuff_OSC_TF[0][t];
   double k_prev = stochBuff_OSC_TF[1][t];
   double d_prev = signalBuff_OSC_TF[1][t];

   if(k_prev < d_prev && k_now > d_now)
      Stochastic_Status[t] = "Buy Signa";
   else if(k_prev > d_prev && k_now < d_now)
      Stochastic_Status[t] =  "Sell Signal";
   else if(k_now > 80 && d_now > 80)
      Stochastic_Status[t] =  "OB_Ehtiait_Nozool";
   else if(k_now < 20 && d_now < 20)
      Stochastic_Status[t] =  "OS_Ehtiait_Souod";
   else
      Stochastic_Status[t] =  "Neutral";
      
      
      // PTL Analysis

      
      
   if(PTL_arrowCol[1] == 0 && PTL_arrowVal[1] != EMPTY_VALUE)
      PTL_signal[t] = "Buy Signa_StartSouod";
   else if(PTL_arrowCol[1] == 1 && PTL_arrowVal[1] != EMPTY_VALUE)
      PTL_signal[t] =  "Sell Signal_StartNozool";
  else
      PTL_signal[t] =  "No_newSignal";
      
   if(PTL_trend[0] == 0)
      PTL_Status[t] =  "PTL_isSouodi";
   else if(PTL_trend[0] == 1)
      PTL_Status[t] =  "PTL_isNozooli";
   else
      PTL_Status[t] =  "PTL_isNeutral";
      
      //------------------------felan END Of TIMEFRAME FOR
                                 }
      //------------------------felan END Of TIMEFRAME FOR
	

	  



      string MultiTF_Status_Header ="MultiTF_Status"+","+
      "Ravand_TF "+"15 min= " +","+ "Ravand_TF "+"1h= " +","+"Ravand_TF "+"4h= " +","+"Ravand_TF "+"D= " +","+ 
      "Ravand_Zstr_TF "+"15 min= " +","+ "Ravand_Zstr_TF "+"1h= " +","+"Ravand_Zstr_TF "+"4h= " +","+"Ravand_Zstr_TF "+"D= " +","+ 
      "PTL_signal "+"15 min= " +","+ "PTL_signal "+"1h= " +","+"PTL_signal "+"4h= " +","+"PTL_signal "+"D= " +","+ 
      "PTL_Status "+"15 min= " +","+ "PTL_Status "+"1h= " +","+"PTL_Status "+"4h= " +","+"PTL_Status "+"D= " +","+ 
      
	   "RSI_Status "+"15 min= " +","+ "RSI_Status "+"1h= " +","+"RSI_Status "+"4h= " +","+"RSI_Status "+"D= " +","+ 
	   "RSI "+"15 min= " +","+ "RSI "+"1h= " +","+"RSI "+"4h= " +","+"RSI "+"D= " +","+ 
	   "MACD_Status "+"15 min= " +","+ "MACD_Status "+"1h= " +","+"MACD_Status "+"4h= " +","+"MACD_Status "+"D= " +","+ 
	   "Stochastic_Status "+"15 min= " +","+ "Stochastic_Status "+"1h= " +","+"Stochastic_Status "+"4h= " +","+"Stochastic_Status "+"D= " +","+ 
	   
      "Supp_Z_TF "+"15 min= " +","+ "Supp_Z_TF "+"1h= " +","+"Supp_Z_TF "+"4h= " +","+"Supp_Z_TF "+"D= " +","+ 
      "Ress_Z_TF "+"15 min= " +","+ "Ress_Z_TF "+"1h= " +","+"Ress_Z_TF "+"4h= " +","+"Ress_Z_TF "+"D= " +","+ 
	   "n_bar_Change_ravand_TF "+"15 min= " +","+ "n_bar_Change_ravand_TF "+"1h= " +","+"n_bar_Change_ravand_TF "+"4h= " +","+"n_bar_Change_ravand_TF "+"D= " +","+ 
	   "n_bar_Change_ravand_Zstr "+"15 min= " +","+ "n_bar_Change_ravand_Zstr "+"1h= " +","+"n_bar_Change_ravand_Zstr "+"4h= " +","+"n_bar_Change_ravand_Zstr "+"D= " +","+ 
      
	   "now_ravand_perc_TF "+"15 min= " +","+ "now_ravand_perc_TF "+"1h= " +","+"now_ravand_perc_TF "+"4h= " +","+"now_ravand_perc_TF "+"D= " +","+ 
	   "sec_ravand_perc_TF "+"15 min= " +","+ "sec_ravand_perc_TF "+"1h= " +","+"sec_ravand_perc_TF "+"4h= " +","+"sec_ravand_perc_TF "+"D= " +","+ 
	   "third_ravand_perc_TF "+"15 min= " +","+ "third_ravand_perc_TF "+"1h= " +","+"third_ravand_perc_TF "+"4h= " +","+"third_ravand_perc_TF "+"D= " +",";


      string MultiTF_Status ="MultiTF_Status"+","+
      Ravand_TF[0]+","+Ravand_TF[1]+","+Ravand_TF[2]+","+Ravand_TF[3]+","+
      Ravand_Zstr_TF[0]+","+Ravand_Zstr_TF[1]+","+Ravand_Zstr_TF[2]+","+Ravand_Zstr_TF[3]+","+
      PTL_signal[0]+","+PTL_signal[1]+","+PTL_signal[2]+","+PTL_signal[3]+","+
      PTL_Status[0]+","+PTL_Status[1]+","+PTL_Status[2]+","+PTL_Status[3]+","+
	   RSI_Status[0]+","+RSI_Status[1]+","+RSI_Status[2]+","+RSI_Status[3]+","+
	   RSI_TF[0][0]+","+RSI_TF[0][1]+","+RSI_TF[0][2]+","+RSI_TF[0][3]+","+
	   MACD_Status[0]+","+MACD_Status[1]+","+MACD_Status[2]+","+MACD_Status[3]+","+
	   Stochastic_Status[0]+","+Stochastic_Status[1]+","+Stochastic_Status[2]+","+Stochastic_Status[3]+","+
      Supp_Z_TF[0]+","+Supp_Z_TF[1]+","+Supp_Z_TF[2]+","+Supp_Z_TF[3]+","+
      Ress_Z_TF[0]+","+Ress_Z_TF[1]+","+Ress_Z_TF[2]+","+Ress_Z_TF[3]+","+
	   n_bar_Change_ravand_TF[0]+","+n_bar_Change_ravand_TF[1]+","+n_bar_Change_ravand_TF[2]+","+n_bar_Change_ravand_TF[3]+","+
	   n_bar_Change_ravand_Zstr_TF[0]+","+n_bar_Change_ravand_Zstr_TF[1]+","+n_bar_Change_ravand_Zstr_TF[2]+","+n_bar_Change_ravand_Zstr_TF[3]+","+
      now_ravand_perc_TF[0]+","+now_ravand_perc_TF[1]+","+now_ravand_perc_TF[2]+","+now_ravand_perc_TF[3]+","+
      sec_ravand_perc_TF[0]+","+sec_ravand_perc_TF[1]+","+sec_ravand_perc_TF[2]+","+sec_ravand_perc_TF[3]+","+
      third_ravand_perc_TF[0]+","+third_ravand_perc_TF[1]+","+third_ravand_perc_TF[2]+","+third_ravand_perc_TF[3]+",";
	  
	  
	  

      
      
      

      //--------------------------------------------------
      // Print("Before candle Parameters  ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
      //----------------------------------candle Parameters------------------------------//
      //if(mrate[0].open==0 || mrate[0].low==0)
      //  {
      //   continue;
      //  }
      // saye ha ba farz + boodan bar
      double saye_up=(mrate[0].high-mrate[0].close/mrate[0].open)*100;
      double saye_down=(mrate[0].open-mrate[0].low/mrate[0].open)*100;
      double Bar_high=((mrate[0].high-mrate[0].low)/mrate[0].low)*100;
      double Bar_high_noshadow=((mrate[0].close-mrate[0].open)/mrate[0].open)*100;
      double Bar_high_Price=MathAbs((mrate[0].high-mrate[0].low));
      double Bar_body_Price=MathAbs((mrate[0].open-mrate[0].close));
      //double shade_low=0;

//
//-------------------vaziat kol bazar-------------------------------
         P3_Bazar_shakhes[i]=//m_symbols_array[shakhesIndex]
                             //+","+mrate_shakhes_D[0].time             //+",Last Bar Time,"+mrate[shakhesIndex].time
                             //+","+Ravand_symbol[shakhesIndex]
                             Posit_symb_cunt_disp
                             +","+Nega_symb_cunt_disp
                             +","+Zero_symb_cunt_disp
                             +","+Up_mos3_symb_cunt_disp
                             +","+Low_manf3_symb_cunt_disp;

      //----------------------------------------------
      //Print("Before my Analysis ",m_symbols_array[i],
      //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);      
      ///--------------------------------------------
      ///-------------------------------- my Analysis
      ///--------------------------------------------
      double deltaRSI,AV_5_OBV,deltaOBV,AV_5_Vol,delta_Vol,deltaOBV_2;
      if(RSI[1]==0)
        {
         Print("symbol",m_symbols_array[i],"RSI buffer  =0!");
         // break;
        }

      //AV_5_OBV=MathAbs(OBVBuffer[0]+OBVBuffer[1]+OBVBuffer[2]+OBVBuffer[3]+OBVBuffer[4])/5;
      AV_5_Vol=(mrate[1].tick_volume+mrate[2].tick_volume+mrate[3].tick_volume+mrate[4].tick_volume+mrate[5].tick_volume)/5;
      if( AV_5_Vol==0 || mrate[0].open==0)
        {
         continue;
        }
      deltaRSI=(RSI[0]-RSI[1]);
      //deltaOBV=(OBVBuffer[0]-OBVBuffer[1])/AV_5_OBV;
      //deltaOBV_2=(OBVBuffer[1]-OBVBuffer[2])/AV_5_OBV;
      delta_Vol=mrate[0].tick_volume/AV_5_Vol;

      double Day_price_change=((MathAbs(mrate[0].open-mrate[0].close)/mrate[0].open)*100);

      //--------------------------------------------------------------------------------
      // Print("Before Jump in Vol Signal ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);    
      //-----------------------------Jump in Vol Signal--------------------

      if(Ravand=="Ravand Nozooli" || Ravand=="Ravand Soodi")
        {
         for(int j=0;j<=n_bar_Change_ravand[i];j++)
           {
            if(mrate[j].tick_volume>2*Ave_100_vol[i])
              {
               signalVol[i]=true;
               signalVol_nBar[i]=j;
               break;
              }
            else
              {
               signalVol[i]=false;
               signalVol_nBar[i]=0;
              }
           }
        }
      if(Ravand=="Ravand_Transient Soodi To Nozooli" || Ravand=="Ravand_Transient Nozooli To Soodi")
        {
         for(int j=0;j<=Chg_RavandBarNumb2nd;j++)
           {
            if(mrate[j].tick_volume>2*Ave_100_vol[i])
              {
               signalVol[i]=true;
               signalVol_nBar[i]=j;
               break;
              }
            else
              {
               signalVol[i]=false;
               signalVol_nBar[i]=0;
              }
           }
        }


        
       
      //-------------------------------

      //--------------------------------- vagarayi------------------
      // Vagarayi mosbat signal kharid (ravand nozooli but hajm soodi)
      if(Ravand=="Ravand Nozooli" && 
         mrate[2].tick_volume<mrate[1].tick_volume && mrate[1].tick_volume<mrate[0].tick_volume
         )//&& mrate[2].tick_volume>Hajm_Mabna[i]
        {
         vagarayiMosbat[i]=true;
        }
      else
        {
         vagarayiMosbat[i]=false;
        }
      // Vagarayi manfi signal foroosh (ravand soodi but hajm nozooli)
      if(Ravand=="Ravand Soodi" && 
         mrate[2].tick_volume>mrate[1].tick_volume && mrate[1].tick_volume>mrate[0].tick_volume
         )//&& mrate[0].tick_volume>Hajm_Mabna[i]
        {
         vagarayiManfi[i]=true;
        }
      else
        {
         vagarayiManfi[i]=false;
        }
        
        
      //--------------------------------------------------
      // Print("Before Writh in File last day parameter  ",m_symbols_array[i],
      // " tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
      //-----------------------------------Write in File last day parameter of Symbols For offline Analysis---------------//
      //int SannT=Sanaat[i];

      double ResToSup_percent=0;
      double LastP_To_Sup_percent=0;
      double RessToLast_percent=(((Ress_Z-last_price)/last_price)*100);

      double ResToSup_percent_Zstr=0;
      double LastP_To_Sup_percent_Zstr=0;
      double RessToLast_percent_Zstr=(((Ress_Z_Zstr-last_price)/last_price)*100);

      if(Supp_Z!=0 && Supp_Z_Zstr!=0)
        {
         ResToSup_percent=(((Ress_Z-Supp_Z)/Supp_Z)*100);
         LastP_To_Sup_percent=(((last_price-Supp_Z)/Supp_Z)*100);

         ResToSup_percent_Zstr=(((Ress_Z_Zstr-Supp_Z_Zstr)/Supp_Z_Zstr)*100);
         LastP_To_Sup_percent_Zstr=(((last_price-Supp_Z_Zstr)/Supp_Z_Zstr)*100);
        }

      // baz khani tick price info     
      SymbolInfoTick(m_symbols_array[i],latest_price); //daryaf etelaat gheymati namad, dar in lahze (bare 2) va rad kardan namad dar soorat sefr boodan last Price 
      last_price=latest_price.ask;
      
      //--------------------------------------------------
      //Print("Before After This Not Work In Close market  ",m_symbols_array[i],
      //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
      //-------------------------------------------------------------------------------------------------------------------//
      //-------------------------------------------After This Not Work In Close market or Symbol With Last Price =0-------------//
      //-------------------------------------------------------------------------------------------------------------------//

      //------------------------ define Last_price Va rad kardan namad agar 0 bood(bazar baste ya hanooz moamele nashode, ya namad basteh)
      SymbolInfoTick(m_symbols_array[i],latest_price); //daryaf etelaat gheymati namad, dar in lahze (bare 2) va rad kardan namad dar soorat sefr boodan last Price 
      last_price=latest_price.ask;
      if(last_price==0)
        {
         continue;  // Analysis depth market ro az dast midam va tashkhis saf kharid o foroosh
        }

      percent_Price_now[i]=((last_price-Last_day_CP_Symbol[i])/last_price)*100;
      perc_Price_15day[i]=((last_price-mrate[15].close)/last_price)*100;

      //------------------------------------------counter of number of change + or - in buy day
      if(LastPercent[i]<0 && percent_Price_now[i]>=0)  // it needs to run All Bazar Time
        {
         counter_Chng_Percent[i]=counter_Chng_Percent[i]+1;
        }

      LastPercent[i]=percent_Price_now[i];
      
      if(i>=symbol_size) //----- inja dige shakhes ha ra rad mikonad 
        {
         continue;
        }
      LastTickAsk[i]=latest_price.ask;
      LastTickBid[i]=latest_price.bid;

      string P6_Win_B_or_S=Vol_win_sell[i]+","+tick_cunt_win_sell[i]+","+Ave_Perc_P_win_sell[i]
                             +","+Vol_win_buy[i]+","+tick_cunt_win_buy[i]+","+Ave_Perc_P_win_buy[i];

   ///-----------------------------------------------
   //--- Do we have positions opened already?
   
   ///-----------------------------------------------
   
      bool Buy_opened=false;  // variable to hold the result of Buy opened position
      bool Sell_opened=false; // variables to hold the result of Sell opened position
   
      if(PositionSelect(m_symbols_array[i])==true) // we have an opened position
        {
         if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
           {
            Buy_opened=true;  //It is a Buy
           }
         else if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
           {
            Sell_opened=true; // It is a Sell
           }
        }
        
        
//----------------- counting total buy and sell
if(PositionsTotal()>0)
  {
   int buys=0,sells=0;
   CalculateAllPositions(buys,sells);
   
   total_Current_BUY=buys;
   total_Current_SELL=sells;
  }
  
  //---------------- avreage price of each type of positions 
  
  
  

//   ///-----------------------------------------------
//   ///----------------------BUY----------------------
//   ///-----------------------------------------------
         int count_bar_LBuy=3;                 //???????????
         if(Symbol_Last_Buy_Time[i][1]!=NULL)       //???????????
              {
                 //-number of Bars after Last Sell      
                 count_bar_LBuy=iBarShift(m_symbols_array[i],PERIOD_CURRENT, StringToTime(Symbol_Last_Buy_Time[i][1]));
                 //Print(m_symbols_array[i],"numb bar from Last Sell",count_bar_LSell)   ;   
              }
//if((CTi_Stru.hour<17 || (CTi_Stru.hour==16 && CTi_Stru.min<45) ))
//  {
// tommorow_Buy[i]==false;
//  }
//----------------------------Not Work After this, if we have Buy position for i symbol---------------//
         if(!Buy_opened ) /////  && !Sell_opened
         //&& (CTi_Stru.hour>=17 || (CTi_Stru.hour==16 && CTi_Stru.min>=45) )
         {
         // Por kardan struct latest_price va estefade 
         double ask_price_=latest_price.ask;
         double bid_price=latest_price.bid;
         double ask_ta_bid_darsad=0;
         if(bid_price!=0)
           {
         ask_ta_bid_darsad=((ask_price_-bid_price)/bid_price)*100;         
           }
        double SarBeSarPrice=ask_price_*1.015;
         //------------------------ Buy Order Inputs -------------------//   
         ZeroMemory(mrequest);
         mrequest.action = TRADE_ACTION_DEAL;   // immediate order execution
         mrequest.sl = 0;                       // Stop Loss
         mrequest.tp = 0;                       // Take Profit
         mrequest.symbol = m_symbols_array[i];  // currency pair
         mrequest.price = NormalizeDouble(latest_price.ask,_Digits);           // latest ask price
         double Vol_req=1;
         //if(Vol_req<1)
         //  {
         //   Vol_req=1;
         //  }
         mrequest.volume = NormalizeDouble(Vol_req*0.01,2);             // number of lots to trade
         mrequest.magic = EA_Magic;                                            // Order Magic Number
         mrequest.type = ORDER_TYPE_BUY;                                       // Buy Order
         //mrequest.type_filling = SYMBOL_FILLING_IOC;                           // Order execution type
         mrequest.deviation=0;   
         

         
                                                   // Deviation from current price
//-----------Account--------------
//--- Show all the information available from the function AccountInfoDouble()
         double BALANCE = AccountInfoDouble(ACCOUNT_BALANCE);
         double CREDIT =  AccountInfoDouble(ACCOUNT_CREDIT);
         double PROFIT =  AccountInfoDouble(ACCOUNT_PROFIT);
         double EQUITY =  AccountInfoDouble(ACCOUNT_EQUITY);
         double MARGIN =  AccountInfoDouble(ACCOUNT_MARGIN);
         double MARGIN_FREE =  AccountInfoDouble(ACCOUNT_MARGIN_FREE);
         double MARGIN_LEVEL = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
         double MARGIN_SO_CALL =AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
         double MARGIN_SO_SO = AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
//---------agar dirooz ejaze kharid sader shodeh, kharid anjam bedeh------------




//tommorow_Buy[i]=true;
          if( tommorow_Buy[i]==true && last_price<500 && MARGIN_FREE>30)// && (CTi_Stru.hour<22 || (CTi_Stru.hour==22 && CTi_Stru.min<=30)
             //date_of_accept_buy[i]!=TimeToString(mrate[0].time,TIME_DATE) && && ((CTi_Stru.hour==9 && CTi_Stru.min>=20) || (CTi_Stru.hour==9 && CTi_Stru.min<=40)) && safKharid[i]==true
            {
            
            if(total_Current_BUY-total_Current_SELL>=1 || (total_Current_SELL==1 && total_Current_BUY==1 ))		// tedad buy ha nahayatan 1 adad bishtar az sell bashad
              {
              tommorow_Buy[i]=false;
              Write_his_buy[i]=false;
              tommorow_Buy_dailyAlert[i]=false;
              continue;
              }

          
          //M1_400_BefBuyTime[i]=mrate_HH1[399].time; //bade sell pak beshe vase har namad
          mrequest.comment=Buy_Strategy[i];
          OrderSend(mrequest,mresult);

         // get the result code
         if(mresult.retcode==10009 || mresult.retcode==10008) //Request is completed or order placed
              {
               Alert("A Buy order has been successfully placed with Ticket#:",mresult.order,"!!");
              tommorow_Buy[i]=false;
              Write_his_buy[i]=false;
              tommorow_Buy_dailyAlert[i]=false;
              continue;
              }
         else
              {
               Alert("The Buy order request could not be completed -error:",GetLastError());
               ResetLastError();   
               // age ham kharid nakard dar in lahze baz bayad chack becshe va age sharayet ok bud kharid beshe
              tommorow_Buy[i]=false;
              Write_his_buy[i]=false;
              tommorow_Buy_dailyAlert[i]=false;
              } 
            }
 //---------------------------------------------------            
       //Print("Before BUY PARAMETER FOR ALL BUYS STRATEGY  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
//-----------------------------------BUY PARAMETER FOR ALL BUYS STRATEGY---------------------------//
      if(last_price==0 || Supp_Z==0 )
     {
     Supp_Z=1;
      //continue;
     }
            //int SannT=Sanaat[i];
            double DL1vol=mrate[1].tick_volume;
            double DL2vol=mrate[2].tick_volume;
            double DL3vol=mrate[3].tick_volume;
            
            
            double RessRavand=((Chg_RavandInPrice2nd-Chg_RavandInPrice4nd)/Chg_RavandInPrice2nd)*100;

//---------------------------------------------------            
//Print("Before tatmam halat haye buy  ",m_symbols_array[i],
//" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
//-------------------------------------------------------------------------------//
//-----------------------------  tatmam halat haye buy --------------------------//
//-----------------------------------Buy Signal For channel 25-8-97 -------------------------------//

// -------------------signals for day parameter--------

//-------------------------------- Posit Signals --------------------------------


if(RSI[0]<55 && signalVol[i]==true && signalVol_pos_Sig[i]==false )
  {
   Posit_signals[i]=Posit_signals[i]+",signalVol_Pos";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   signalVol_pos_Sig[i]=true;
  }

if(counter_Chng_Percent[i]>4 && Chng_Percent_Sig[i]==false )
  {
   Posit_signals[i]=Posit_signals[i]+",counter_Chng_Percent_more_5";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   Chng_Percent_Sig[i]=true;
  }
  
if(Bar_high_noshadow>3 && High_Bar_Mosbat_Sig[i]==false )
  {
   Posit_signals[i]=Posit_signals[i]+",High_Bar_Mosbat ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   High_Bar_Mosbat_Sig[i]=true;
  }
  
if((((last_price-Supp_Z)/Supp_Z)*100)<4 && (((last_price-Supp_Z)/Supp_Z)*100)>-4 && near_sup_Sig[i]==false)
  {
   Posit_signals[i]=Posit_signals[i]+",near_sup ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   near_sup_Sig[i]=true;
  }
 
if(now_ravand_perc_TF[3]<-13 && more_13_eslah_Sig[i]==false )        //---------------- for timeframe (should be checked)-------
  {
   Posit_signals[i]=Posit_signals[i]+",more -13 eslah ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   more_13_eslah_Sig[i]=true;
  }
  
if(sec_ravand_perc_TF[3]<-17 && now_ravand_perc_TF[3]<8 && LastRav_Sig[i]==false)//---------------- for timeframe (should be checked)-------
  {
   Posit_signals[i]=Posit_signals[i]+",LastRav<-17and now <8 ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   LastRav_Sig[i]=true;
  }
  
if((deltaMACD1_TF[3]>deltaMACD2_TF[3] && deltaMACD2_TF[3]>deltaMACD3_TF[3]) && MACD_pos_Sig[i]==false) //---------------- for timeframe (should be checked)-------
  {
   Posit_signals[i]=Posit_signals[i]+",deltaMACD_mosbat ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   MACD_pos_Sig[i]=true;
  }

if(stochBuff_OSC[0]>signalBuff_OSC[0] && stochBuff_OSC[1]<25 && OSC_zir25_Soodi_Sig[i]==false)
  {
   Posit_signals[i]=Posit_signals[i]+",OSC_zir25_Soodi ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   OSC_zir25_Soodi_Sig[i]=true;
  }

if(stochBuff_OSC[0]>signalBuff_OSC[1] && signalBuff_OSC[1]>=stochBuff_OSC[1] && Shekast_OSC_mosbat_Sig[i]==false)
  {
   Posit_signals[i]=Posit_signals[i]+",Shekast_OSC_mosbat ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   Shekast_OSC_mosbat_Sig[i]=true;
  }
  

if(percent_Price_now[i]!=0)
 {
      if((((percent_Price_now[i]-percent_Price_Cl_day[i])/percent_Price_now[i])*100)>2 
      && (CTi_Stru.hour==12 && CTi_Stru.min>15) && Clpric_to_lastP_pos_Sig[i]==false)
        {
         Posit_signals[i]=Posit_signals[i]+",Clpric_<2%KamtarAz_LastPric";
         numb_Posit_sig[i]=numb_Posit_sig[i]+1;
         Clpric_to_lastP_pos_Sig[i]=true;
        }
        
//-------------------------------- Neg Signals --------------------------------

      if((((percent_Price_now[i]-percent_Price_Cl_day[i])/percent_Price_now[i])*100)<-2 
      && (CTi_Stru.hour==12 && CTi_Stru.min>15) && Clpric_to_lastP_neg_Sig[i]==false )
        {
         Nega_signals[i]=Nega_signals[i]+",Clpric_<2%BishtarAz_LastPric";
         numb_Nega_sig[i]=numb_Nega_sig[i]+1;
         Clpric_to_lastP_neg_Sig[i]=true;
        }
}


if(RSI[0]>65 && signalVol[i]==true && signalVol_neg_Sig[i]==false )
  {
   Nega_signals[i]=Nega_signals[i]+",signalVol_neg";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   signalVol_neg_Sig[i]=true;
  }

if(Bar_high_noshadow<-3 && RSI[0]>65 && Ravand=="Ravand Soodi" && High_Bar_Manfi_Sig[i]==false)
  {
   Nega_signals[i]=Nega_signals[i]+",High_Bar_Manfi ";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   High_Bar_Manfi_Sig[i]=true;
  }

if((((Ress_Z-last_price)/last_price)*100)<6 && (((Ress_Z-last_price)/last_price)*100)>-2 && near_ress_Sig[i]==false)
  {
   Nega_signals[i]=Nega_signals[i]+",near_ress";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   near_ress_Sig[i]=true;
  }

if(stochBuff_OSC[0]<signalBuff_OSC[1] && signalBuff_OSC[1]<=stochBuff_OSC[1] && Shekast_OSC_manfi_Sig[i]==false)
  {
   Nega_signals[i]=Nega_signals[i]+",Shekast_OSC_manfi ";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   Shekast_OSC_manfi_Sig[i]=true;
  }

sum_numb_Posit_sig=sum_numb_Posit_sig+numb_Posit_sig[i];
sum_numb_Nega_sig=sum_numb_Nega_sig+numb_Nega_sig[i];
checked_symbol_count=checked_symbol_count+1;


string P_Neg_Sig=Clpric_to_lastP_neg_Sig[i]+","+signalVol_neg_Sig[i]
+","+High_Bar_Manfi_Sig[i]+","+near_ress_Sig[i]+","+Shekast_OSC_manfi_Sig[i];

string P_Posit_sig=Chng_Percent_Sig[i]
+","+High_Bar_Mosbat_Sig[i]+","+near_sup_Sig[i]+","+more_13_eslah_Sig[i]
+","+LastRav_Sig[i]+","+MACD_pos_Sig[i]+","+OSC_zir25_Soodi_Sig[i]+","+Shekast_OSC_mosbat_Sig[i]+","+Clpric_to_lastP_pos_Sig[i];


//---------------------------------- Dayli parameter string creating----------------------------

string P1_symbol=m_symbols_array[i]+","+mrate[0].time;
string P2_BuyStr="";
		//string P5_Sahm_bonyad=Tot_cost_of_symb+","+P_on_E[i]+","+eps[i]+","+(Ave_Cost_3M[i]/10000000)+","+rotbe_naghd[i]+","+Ave_Price_naghd;
string P7_Indicators=RSI[0]
             +","+stochBuff_OSC[0]
             +","+signalBuff_OSC[0]
             +","+(stochBuff_OSC[0]-signalBuff_OSC[0])
             +","+deltaMACD1_TF[3]
             +","+deltaMACD2_TF[3]
             +","+deltaMACD3_TF[3];                      //---------------- for timeframe (should be checked)-------
             //+","+IndBuffer[0]+","+SigBuffer[1]
             //+","+last_buy_oscSig_day+","+last_sell_oscSig_day;
string P8_RavandP=Ravand+","+n_bar_Change_ravand[i]+","+Ravand_Zstr+","+n_bar_Change_ravand_Zstr[i]
       +","+now_ravand_perc_TF[3]+","+sec_ravand_perc_TF[3]+","+third_ravand_perc_TF[3];     //---------------- for timeframe (should be checked)-------
string P9_Sup_and_RessP=(((last_price-Supp_Z)/Supp_Z)*100)+","+(((Ress_Z-last_price)/last_price)*100);
string P10_just_dayP=percent_Price_now[i]+","+percent_Price_Cl_day[i]+","+Bar_high+","+Bar_high_noshadow
                     +","+counter_Chng_Percent[i]+","+"NOTHING"+","+mrate[0].tick_volume/Ave_100_vol[i];
//string P101_saf_parameter=safKharid[i]+","+count_safKharid_day[i]+","+last_safKharid_start[i]+","+last_safKharid_break[i]
//      +","+safFrush[i]+","+count_safFrush_day[i]+","+last_safFrush_start[i]+","+last_safFrush_break[i];
string P11_signals=signalVol[i]+","+signalVol_nBar[i]+","+vagarayiManfi[i]+","+vagarayiMosbat[i];
string P12_candels=Candle_type+","+SymbolCandel[i][1]+","+SymbolCandel[i][2]+","+SymbolCandel[i][3]+","+SymbolCandel[i][4];
string P13_extraP=last_price+","+Ress_Z+","+Ress_Z_STR+","+Supp_Z+","+Supp_Z_STR+","+"Last_Souod_Or_Nozol_perc"
             +","+mrate[1].tick_volume+","+mrate[0].tick_volume+","+mrate[1].tick_volume+","+mrate[0].tick_volume
             +","+Ave_100_Price_Perc[i]+","+deltaRSI
             +","+ResToSup_percent_Zstr+","+LastP_To_Sup_percent_Zstr
             +","+RessToLast_percent_Zstr
             +","+Ress_Z_Zstr+","+Ress_Z_STR_Zstr+","+Supp_Z_Zstr+","+Supp_Z_STR_Zstr
             +","+"Last_Souod_Or_Nozol_perc_Zstr"
             +","+this_day_CP[i]+","+Last_day_CP_Symbol[i]
             +","+symbol_des[i]+","+symbol_path[i];
string P14_FastCheck=numb_growD_5day[i]+","+numb_growD_10day[i]+","+numb_growD_20day[i]+","+numb_Posit_sig[i]
+","+numb_Nega_sig[i]+",Pos_Sigs :,"+P_Posit_sig+",Neg_Sigs :,"+P_Neg_Sig;

string P15__ASK_to_bid_Perc=ask_ta_bid_darsad;

//----------------- Python input file created and RUN  ----------


      if (PythonCreatedFileTime!=Ctime && FirstCreatedPythonFile==true)
      {
      count_M1_PythonFileCreated=iBarShift(m_symbols_array[i],PERIOD_M1, PythonCreatedFileTime);
      }
      
            if (FirstCreatedPythonFile == false || count_M1_PythonFileCreated > 1)
              {
               // حذف فایل قبلی از پوشه مشترک
               FileDelete("inputFile_Python.csv", FILE_COMMON);
            
               // ایجاد فایل جدید در پوشه مشترک
               int file_handle_inputPython = FileOpen("inputFile_Python.csv", FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON);
               if(file_handle_inputPython == INVALID_HANDLE)
                 {
                  PrintFormat("❌ Failed to open %s file, Error code = %d", "inputFile_Python.csv", GetLastError());
                 }
               else
                 {
                  FileWriteString(file_handle_inputPython, MultiTF_Status_Header + "\r\n" + MultiTF_Status);
                  FileClose(file_handle_inputPython);
                  Print("✅ inputFile_Python.csv created in Common\\Files");
                 }
            
               PythonCreatedFileTime = Ctime;
               FirstCreatedPythonFile = true;
               count_M1_PythonFileCreated = 0;
            
               // کپی فایل در همان پوشه مشترک
               FileCopy("inputFile_Python.csv", FILE_COMMON, "inputFile_Python2.csv", FILE_COMMON);
              }

        
             // 3. انتظار برای تولید خروجی
             Sleep(20000);  // 2 ثانیه صبر کن (می‌تونی بهبود بدی با چک کردن وجود فایل)
             //    FileDelete("inputFile_Python.csv");
             
             // 4. خواندن خروجی
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
            double p_15min = loader.GetProfitProbByTag("15minBuy");
            double n_15min = loader.GetNeutralProbByTag("15minBuy");
            double l_15min = loader.GetLossProbByTag("15minBuy");
            
            double p_1h = loader.GetProfitProbByTag("1hBuy");
            double n_1h = loader.GetNeutralProbByTag("1hBuy");
            double l_1h = loader.GetLossProbByTag("1hBuy");
         
            double p_2h = loader.GetProfitProbByTag("2hBuy");
            double n_2h = loader.GetNeutralProbByTag("2hBuy");
            double l_2h = loader.GetLossProbByTag("2hBuy");
         
            double p_3h = loader.GetProfitProbByTag("3hBuy");
            double n_3h = loader.GetNeutralProbByTag("3hBuy");
            double l_3h = loader.GetLossProbByTag("3hBuy");
            
            double p_4h = loader.GetProfitProbByTag("4hBuy");
            double n_4h = loader.GetNeutralProbByTag("4hBuy");
            double l_4h = loader.GetLossProbByTag("4hBuy");
            
            double p_1d = loader.GetProfitProbByTag("1DBuy");
            double n_1d = loader.GetNeutralProbByTag("1DBuy");
            double l_1d = loader.GetLossProbByTag("1DBuy");
            
            FileDelete("prediction_15minBuy.txt", FILE_COMMON);
            FileDelete("prediction_1hBuy.txt", FILE_COMMON);
            FileDelete("prediction_2hBuy.txt", FILE_COMMON);
            FileDelete("prediction_3hBuy.txt", FILE_COMMON);
            FileDelete("prediction_4hBuy.txt", FILE_COMMON);
            FileDelete("prediction_1DBuy.txt", FILE_COMMON);

   
//////// ------------------- Buy_Strategy_4 signals for day parameter
double Bar_AV_M0=(mrate_HH1[0].close+mrate_HH1[0].open)/2;
double Bar_AV_M1=(mrate_HH1[1].close+mrate_HH1[1].open)/2;
double Bar_AV_M2=(mrate_HH1[2].close+mrate_HH1[2].open)/2;
double Bar_AV_M3=(mrate_HH1[3].close+mrate_HH1[3].open)/2;
double Bar_AV_M4=(mrate_HH1[4].close+mrate_HH1[4].open)/2;
double shib_AV_M4_M3=Bar_AV_M3-Bar_AV_M4;
double shib_AV_M3_M2=Bar_AV_M2-Bar_AV_M3;
double shib_AV_M2_M1=Bar_AV_M1-Bar_AV_M2;
double shib_AV_M1_M0=Bar_AV_M0-Bar_AV_M1;





         if( p_1d>0.95 &&  p_4h>0.6 // && p_1h>0.8 && p_2h>0.9 //!Buy_opened
//		 (PTL_signal[0]=="Buy Signa_StartSouod" && PTL_Status[1]=="PTL_isSouodi" && PTL_Status[2]=="PTL_isSouodi" && PTL_Status[3]=="PTL_isSouodi") ||
//         (PTL_signal[0]=="Buy Signa_StartSouod" && PTL_Status[1]=="PTL_isSouodi" && PTL_Status[2]=="PTL_isSouodi") ||
//         (PTL_signal[0]=="Buy Signa_StartSouod" && PTL_Status[1]=="PTL_isSouodi" && PTL_Status[3]=="PTL_isSouodi") 
         //Ravand=="Ravand Soodi" && ZigZag_Buffer[0]!=0 //&& shib_ravand[0]>=0.7*Ave_shib_rav_mosbat
        //&& ((Chg_RavandInPrice2nd_Zstr-last_price)/last_price)*100>=0.5*Ave__UptoDo_mosb_Zst
         //SAR[1]>last_price && SAR[0]<last_price
         //shib_AV_M4_M3<0 && shib_AV_M3_M2<0 && shib_AV_M2_M1>0 && shib_AV_M1_M0>0 && shib_AV_M1_M0>shib_AV_M2_M1 //Bar_AV_M1>Bar_AV_M2 && mrate_HH1[0].open>Bar_AV_M1 && mrate_HH1[0].close>mrate_HH1[0].open
         //&& CTi_Stru.hour<22
         
         //(CTi_Stru.hour==22 && CTi_Stru.min>29) //((mrate[0].close-mrate[0].open)/mrate[0].open*100)<(-1.5*Ave_100_Price_Perc[i])
          //Ravand=="Ravand Soodi" && Ravand_Zstr=="Ravand Soodi" 
         //&& stochBuff_OSC[0]>35
         // && stochBuff_OSC[0]<signalBuff_OSC[0] && stochBuff_OSC[0]-stochBuff_OSC[1]<-2 && stochBuff_OSC[2]>stochBuff_OSC[1]
         //&& RSI[0]-RSI[1]<=-2 &&RSI[1]<RSI[2]
          //&& ((stochBuff_OSC[2]-stochBuff_OSC[1])<=(stochBuff_OSC[1]-stochBuff_OSC[0]))
          //&& n_bar_Change_ravand[i]<=5
         //stochBuff_OSC[0]>signalBuff_OSC[0] && stochBuff_OSC[0]-stochBuff_OSC[1]>2  //
          //&& RSI[0]-RSI[1]>=22
          //&& percent_Price_now[i]<2
          //&& RSI[0]<70
           //&& RSI[1]>RSI[2]  //&& RSI[0] >50 && RSI[2]>RSI[3]
          //&& MACDBuffer[0]>MACDBuffer[1] //&& MACDSignal[0]>MACDSignal[1]  //&& signalBuff_OSC[0]>=stochBuff_OSC[0]
          //&&((deltaMACD1>deltaMACD2)) 
          //&& Strong_Signal_From_zigzag[i]==false 

          
          
//          
           //&&((mrate_HH1[1].close-mrate_HH1[1].open)/mrate_HH1[1].open)*100>0.1
//          && ((mrate_HH1[2].close-mrate_HH1[2].open)/mrate_HH1[2].open)*100>0.2
//          && ((mrate_HH1[3].close-mrate_HH1[3].open)/mrate_HH1[3].open)*100>0.2
          //Ravand=="Ravand Soodi"  // && stochBuff_OSC[0]>35
          //&& stochBuff_OSC[0]<35 && Ravand_Zstr=="Ravand Soodi"
            )//     && && CTi_Stru.hour<22 (((last_price-Supp_Z)/Supp_Z)*100)>=-7           && mrate[0].tick_volume>(1.2*((mrate[1].tick_volume+ mrate[2].tick_volume)/2))   && mrate[0].tick_volume>(1.2*((mrate[1].tick_volume+ mrate[2].tick_volume)/2))  && (((last_price-Supp_Z)/Supp_Z)*100)<=7
           {
              
 //             if((Symbol_Last_Buy_Time[i][1]==TimeToString(Ctime,TIME_DATE) && Symbol_Last_Buy_Type[i][1]=="ZARAR"))
 //               {
 //                continue; //adam kharid dor sorati ke emrooz kharid manfi dashtam
 //               }
              
               tommorow_Buy_dailyAlert[i]= true;

               Buy_Strategy_dailyAlert[i]="Buy_Strategy_1";
                
          Alert("Strategy 1 Buy_accept"+m_symbols_array[i],Email_Buy_Parameters[i]);
          tommorow_Buy[i]= true;
          //Print("Time_date",TimeToString(mrate[0].time,TIME_DATE));
          date_of_accept_buy[i]=TimeToString(mrate[0].time,TIME_DATE);

             Buy_Strategy[i]=Buy_Strategy_dailyAlert[i];//"Buy_Strategy_1"
///---------
            P2_BuyStr=Buy_Strategy_dailyAlert[i]+","+tommorow_Buy_dailyAlert[i];
			
            Last_Day_Symbol_parameters[i]=
            P1_symbol
            +","+P2_BuyStr
            +",Vazeiat bazar va shakhes,"
            +P3_Bazar_shakhes[i]
            //+","+P6_Win_B_or_S
            +","+P7_Indicators
            +","+P8_RavandP
            +","+P9_Sup_and_RessP
            +","+P10_just_dayP
            +","+P11_signals
            //+","+P12_candels
            //+",Extra param"
            //+","+P13_extraP
            //+",FastCheck"
            //+","+P14_FastCheck
			+","+P15__ASK_to_bid_Perc
			+","+MultiTF_Status;


             DayBef_Buy_parameters[i]=Last_Day_Symbol_parameters[i];
                
                
               }



//------------------------------------------- Write day Parameter File
//-------------------- make parameter daily----------------- 

P2_BuyStr=Buy_Strategy_dailyAlert[i]+","+tommorow_Buy_dailyAlert[i];

      if(i<symbol_size)
        {
         Last_Day_Symbol_parameters[i]=
         P1_symbol
         +","+P2_BuyStr
         +",Vazeiat bazar va shakhes,"
         +P3_Bazar_shakhes[i]
         
         //+","+P6_Win_B_or_S
         +","+P7_Indicators
         +","+P8_RavandP
         +","+P9_Sup_and_RessP
         +","+P10_just_dayP
         
         +","+P11_signals;
         //+","+P12_candels
         //+",Extra param"
         //+","+P13_extraP
         //+",FastCheck"
         //+","+P14_FastCheck;
        }


         
        if(first_open_file1==false)
        {
         FileSeek(file_handle_2,0,SEEK_END);
         FileWrite(file_handle_2,Last_Day_Symbol_parameters[i]); 
		 
         for(int j=0;j<size_Portfolio;j++)
           {
           if(m_symbols_array_Portfolio[j]==m_symbols_array[i])
             {
               FileSeek(file_handle_4,0,SEEK_END);
               FileWrite(file_handle_4,Last_Day_Symbol_parameters[i]); 
             }
           }
        }



      


//            }

////-----------------------------------End of Buy Strategis -------------------------------//

	}
  
////----------------------------------------------End Of Buys--------------------------------------//

  if(PositionsTotal()>0)
  {
   int buys=0,sells=0;
   CalculateAllPositions(buys,sells);
   
   total_Current_BUY=buys;
   total_Current_SELL=sells;
  }
  
//+------------------------------------------------------------------+
//|                                     Start  Sell position                             |
//+------------------------------------------------------------------+
          int count_bar_LSell=3;                 //???????????
         if(Symbol_Last_sell_Time[i][1]!=NULL)       //???????????
              {
                 //-number of Bars after Last Sell      
                 count_bar_LSell=iBarShift(m_symbols_array[i],PERIOD_CURRENT, StringToTime(Symbol_Last_sell_Time[i][1]));
                 //Print(m_symbols_array[i],"numb bar from Last Sell",count_bar_LSell)   ;   
              }
              
//if((CTi_Stru.hour<17 || (CTi_Stru.hour==16 && CTi_Stru.min<45) ))
//  {
// tommorow_sell[i]==false;
//  }
//----------------------------Not Work After this, if we have Buy position for i symbol---------------//
         if( !Sell_opened) //////!Buy_opened &&
         // && (CTi_Stru.hour>=17 || (CTi_Stru.hour==16 && CTi_Stru.min>=45) )
         {
         // Por kardan struct latest_price va estefade 
         double ask_price_=latest_price.ask;
         double bid_price=latest_price.bid;
         double ask_ta_bid_darsad=0;
         if(bid_price!=0)
           {
         ask_ta_bid_darsad=((ask_price_-bid_price)/bid_price)*100;         
           }
        double SarBeSarPrice=ask_price_-(ask_price_*1.015);
         
       //  ----------sell input    
      ZeroMemory(mrequest);
      mrequest.action=TRADE_ACTION_DEAL;                               // immediate order execution
      mrequest.price = NormalizeDouble(latest_price.bid,_Digits);      // latest Bid price
      mrequest.sl =0;                                                  // Stop Loss
      mrequest.tp =0 ;                                                 // Take Profit
      mrequest.symbol = m_symbols_array[i];                            // currency pair
      double Vol_req=1;
      //if(Vol_req<1)
      //  {
      //   Vol_req=1;
      //  }
      mrequest.volume = NormalizeDouble(Vol_req*0.01,2);             // number of lots to trade
      mrequest.magic = EA_Magic;                                       // Order Magic Number
      mrequest.type= ORDER_TYPE_SELL;                                  // Sell Order
      //mrequest.type_filling = SYMBOL_FILLING_IOC ;                     // Order execution type
      mrequest.deviation=100;                                          // Deviation from current price
//         
         
                                                   // Deviation from current price
//-----------Account--------------
//--- Show all the information available from the function AccountInfoDouble()
         double BALANCE = AccountInfoDouble(ACCOUNT_BALANCE);
         double CREDIT =  AccountInfoDouble(ACCOUNT_CREDIT);
         double PROFIT =  AccountInfoDouble(ACCOUNT_PROFIT);
         double EQUITY =  AccountInfoDouble(ACCOUNT_EQUITY);
         double MARGIN =  AccountInfoDouble(ACCOUNT_MARGIN);
         double MARGIN_FREE =  AccountInfoDouble(ACCOUNT_MARGIN_FREE);
         double MARGIN_LEVEL = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
         double MARGIN_SO_CALL =AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
         double MARGIN_SO_SO = AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
//---------agar dirooz ejaze kharid sader shodeh, kharid anjam bedeh------------
//tommorow_Buy[i]=true;


          if( tommorow_sell[i]==true && last_price<500 && MARGIN_FREE>30 // && (CTi_Stru.hour<22 || (CTi_Stru.hour==22 && CTi_Stru.min<=30))
          ) //date_of_accept_buy[i]!=TimeToString(mrate[0].time,TIME_DATE) && && ((CTi_Stru.hour==9 && CTi_Stru.min>=20) || (CTi_Stru.hour==9 && CTi_Stru.min<=40)) && safKharid[i]==true
            {
            
               if( total_Current_SELL-total_Current_BUY>=1 || (total_Current_SELL==1 && total_Current_BUY==1 ) )
              {
              tommorow_sell[i]=false;
              Write_his_buy[i]=false;
              tommorow_sell_dailyAlert[i]=false;
              continue;
              }

          
          //M1_400_BefBuyTime[i]=mrate_HH1[399].time; //bade sell pak beshe vase har namad
          mrequest.comment=Sell_Strategy[i];
          OrderSend(mrequest,mresult);

         // get the result code
         if(mresult.retcode==10009 || mresult.retcode==10008) //Request is completed or order placed
              {
               Alert("A Buy order has been successfully placed with Ticket#:",mresult.order,"!!");
              tommorow_sell[i]=false;
              Write_his_buy[i]=false;
              tommorow_sell_dailyAlert[i]=false;
              continue;
              }
         else
              {
               Alert("The Buy order request could not be completed -error:",GetLastError());
               ResetLastError();      
              tommorow_sell[i]=false;
              Write_his_buy[i]=false;
              tommorow_sell_dailyAlert[i]=false;
              } 
            }
 //---------------------------------------------------            
       //Print("Before BUY PARAMETER FOR ALL BUYS STRATEGY  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
//----------------------------------- PARAMETER FOR ALL sells STRATEGY---------------------------//
      if(last_price==0 || Supp_Z==0 )
     {
     Supp_Z=1;
      //continue;
     }
            //int SannT=Sanaat[i];
            double DL1vol=mrate[1].tick_volume;
            double DL2vol=mrate[2].tick_volume;
            double DL3vol=mrate[3].tick_volume;
            
            
            double RessRavand=((Chg_RavandInPrice2nd-Chg_RavandInPrice4nd)/Chg_RavandInPrice2nd)*100;
//            if(mrate_H1[0].close==0 || mrate_H1[0].low==0 || mrate_H1[1].close==0 || mrate_H1[1].low==0
//               || mrate_H1[2].close==0 || mrate_H1[2].low==0 || mrate_H1[3].close==0 || mrate_H1[3].low==0
//               || mrate_H1[4].close==0 || mrate_H1[4].low==0 )


//---------------------------------------------------            
//Print("Before tatmam halat haye buy  ",m_symbols_array[i],
//" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
//-------------------------------------------------------------------------------//
//-----------------------------  tatmam halat haye sell posi --------------------------//
//-----------------------------------sell pos Signal For channel 25-8-97 -------------------------------//

// -------------------signals for day parameter--------

//-------------------------------- Posit Signals --------------------------------


if(RSI[0]<55 && signalVol[i]==true && signalVol_pos_Sig[i]==false )
  {
   Posit_signals[i]=Posit_signals[i]+",signalVol_Pos";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   signalVol_pos_Sig[i]=true;
  }

if(counter_Chng_Percent[i]>4 && Chng_Percent_Sig[i]==false )
  {
   Posit_signals[i]=Posit_signals[i]+",counter_Chng_Percent_more_5";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   Chng_Percent_Sig[i]=true;
  }
  
if(Bar_high_noshadow>3 && High_Bar_Mosbat_Sig[i]==false )
  {
   Posit_signals[i]=Posit_signals[i]+",High_Bar_Mosbat ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   High_Bar_Mosbat_Sig[i]=true;
  }
  
if((((last_price-Supp_Z)/Supp_Z)*100)<4 && (((last_price-Supp_Z)/Supp_Z)*100)>-4 && near_sup_Sig[i]==false)
  {
   Posit_signals[i]=Posit_signals[i]+",near_sup ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   near_sup_Sig[i]=true;
  }
 
if(now_ravand_perc_TF[3]<-13 && more_13_eslah_Sig[i]==false ) //---------------- for timeframe (should be checked)-------
  {
   Posit_signals[i]=Posit_signals[i]+",more -13 eslah ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   more_13_eslah_Sig[i]=true;
  }
  
if(sec_ravand_perc_TF[3]<-17 && now_ravand_perc_TF[3]<8 && LastRav_Sig[i]==false) //---------------- for timeframe (should be checked)-------
  {
   Posit_signals[i]=Posit_signals[i]+",LastRav<-17and now <8 ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   LastRav_Sig[i]=true;
  }
  
if((deltaMACD1_TF[3]>deltaMACD2_TF[3] && deltaMACD2_TF[3]>deltaMACD3_TF[3]) && MACD_pos_Sig[i]==false)//---------------- for timeframe (should be checked)-------
  {
   Posit_signals[i]=Posit_signals[i]+",deltaMACD_mosbat ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   MACD_pos_Sig[i]=true;
  }

if(stochBuff_OSC[0]>signalBuff_OSC[0] && stochBuff_OSC[1]<25 && OSC_zir25_Soodi_Sig[i]==false)
  {
   Posit_signals[i]=Posit_signals[i]+",OSC_zir25_Soodi ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   OSC_zir25_Soodi_Sig[i]=true;
  }

if(stochBuff_OSC[0]>signalBuff_OSC[1] && signalBuff_OSC[1]>=stochBuff_OSC[1] && Shekast_OSC_mosbat_Sig[i]==false)
  {
   Posit_signals[i]=Posit_signals[i]+",Shekast_OSC_mosbat ";
   numb_Posit_sig[i]=numb_Posit_sig[i]+1;
   Shekast_OSC_mosbat_Sig[i]=true;
  }
  

if(percent_Price_now[i]!=0)
 {
      if((((percent_Price_now[i]-percent_Price_Cl_day[i])/percent_Price_now[i])*100)>2 
      && (CTi_Stru.hour==12 && CTi_Stru.min>15) && Clpric_to_lastP_pos_Sig[i]==false)
        {
         Posit_signals[i]=Posit_signals[i]+",Clpric_<2%KamtarAz_LastPric";
         numb_Posit_sig[i]=numb_Posit_sig[i]+1;
         Clpric_to_lastP_pos_Sig[i]=true;
        }
        
//-------------------------------- Neg Signals --------------------------------

      if((((percent_Price_now[i]-percent_Price_Cl_day[i])/percent_Price_now[i])*100)<-2 
      && (CTi_Stru.hour==12 && CTi_Stru.min>15) && Clpric_to_lastP_neg_Sig[i]==false )
        {
         Nega_signals[i]=Nega_signals[i]+",Clpric_<2%BishtarAz_LastPric";
         numb_Nega_sig[i]=numb_Nega_sig[i]+1;
         Clpric_to_lastP_neg_Sig[i]=true;
        }
}


if(RSI[0]>65 && signalVol[i]==true && signalVol_neg_Sig[i]==false )
  {
   Nega_signals[i]=Nega_signals[i]+",signalVol_neg";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   signalVol_neg_Sig[i]=true;
  }

if(Bar_high_noshadow<-3 && RSI[0]>65 && Ravand=="Ravand Soodi" && High_Bar_Manfi_Sig[i]==false)
  {
   Nega_signals[i]=Nega_signals[i]+",High_Bar_Manfi ";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   High_Bar_Manfi_Sig[i]=true;
  }

if((((Ress_Z-last_price)/last_price)*100)<6 && (((Ress_Z-last_price)/last_price)*100)>-2 && near_ress_Sig[i]==false)
  {
   Nega_signals[i]=Nega_signals[i]+",near_ress";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   near_ress_Sig[i]=true;
  }

if(stochBuff_OSC[0]<signalBuff_OSC[1] && signalBuff_OSC[1]<=stochBuff_OSC[1] && Shekast_OSC_manfi_Sig[i]==false)
  {
   Nega_signals[i]=Nega_signals[i]+",Shekast_OSC_manfi ";
   numb_Nega_sig[i]=numb_Nega_sig[i]+1;
   Shekast_OSC_manfi_Sig[i]=true;
  }

sum_numb_Posit_sig=sum_numb_Posit_sig+numb_Posit_sig[i];
sum_numb_Nega_sig=sum_numb_Nega_sig+numb_Nega_sig[i];
checked_symbol_count=checked_symbol_count+1;


string P_Neg_Sig=Clpric_to_lastP_neg_Sig[i]+","+signalVol_neg_Sig[i]
+","+High_Bar_Manfi_Sig[i]+","+near_ress_Sig[i]+","+Shekast_OSC_manfi_Sig[i];

string P_Posit_sig=Chng_Percent_Sig[i]
+","+High_Bar_Mosbat_Sig[i]+","+near_sup_Sig[i]+","+more_13_eslah_Sig[i]
+","+LastRav_Sig[i]+","+MACD_pos_Sig[i]+","+OSC_zir25_Soodi_Sig[i]+","+Shekast_OSC_mosbat_Sig[i]+","+Clpric_to_lastP_pos_Sig[i];


//---------------------------------- Dayli parameter string creating----------------------------

string P1_symbol=m_symbols_array[i]+","+mrate[0].time;
string P2_BuyStr="";
//string P5_Sahm_bonyad=Tot_cost_of_symb+","+P_on_E[i]+","+eps[i]+","+(Ave_Cost_3M[i]/10000000)+","+rotbe_naghd[i]+","+Ave_Price_naghd;
string P7_Indicators=RSI[0]
             +","+stochBuff_OSC[0]
             +","+signalBuff_OSC[0]
             +","+(stochBuff_OSC[0]-signalBuff_OSC[0])
             +","+deltaMACD1_TF[3]
             +","+deltaMACD2_TF[3]
             +","+deltaMACD3_TF[3];                //---------------- for timeframe (should be checked)-------
             //+","+IndBuffer[0]+","+SigBuffer[1]
             //+","+last_buy_oscSig_day+","+last_sell_oscSig_day;
string P8_RavandP=Ravand+","+n_bar_Change_ravand[i]+","+Ravand_Zstr+","+n_bar_Change_ravand_Zstr[i]
       +","+now_ravand_perc_TF[3]+","+sec_ravand_perc_TF[3]+","+third_ravand_perc_TF[3];  //---------------- for timeframe (should be checked)-------
string P9_Sup_and_RessP=(((last_price-Supp_Z)/Supp_Z)*100)+","+(((Ress_Z-last_price)/last_price)*100);
string P10_just_dayP=percent_Price_now[i]+","+percent_Price_Cl_day[i]+","+Bar_high+","+Bar_high_noshadow
                     +","+counter_Chng_Percent[i]+","+"cost_buy_day"+","+mrate[0].tick_volume/Ave_100_vol[i];
//string P101_saf_parameter=safKharid[i]+","+count_safKharid_day[i]+","+last_safKharid_start[i]+","+last_safKharid_break[i]
//      +","+safFrush[i]+","+count_safFrush_day[i]+","+last_safFrush_start[i]+","+last_safFrush_break[i];
string P11_signals=signalVol[i]+","+signalVol_nBar[i]+","+vagarayiManfi[i]+","+vagarayiMosbat[i];
string P12_candels=Candle_type+","+SymbolCandel[i][1]+","+SymbolCandel[i][2]+","+SymbolCandel[i][3]+","+SymbolCandel[i][4];
string P13_extraP=last_price+","+Ress_Z+","+Ress_Z_STR+","+Supp_Z+","+Supp_Z_STR+","+"Last_Souod_Or_Nozol_perc"
             +","+mrate[1].tick_volume+","+mrate[0].tick_volume+","+mrate[1].tick_volume+","+mrate[0].tick_volume
             +","+Ave_100_Price_Perc[i]+","+deltaRSI
             +","+ResToSup_percent_Zstr+","+LastP_To_Sup_percent_Zstr
             +","+RessToLast_percent_Zstr
             +","+Ress_Z_Zstr+","+Ress_Z_STR_Zstr+","+Supp_Z_Zstr+","+Supp_Z_STR_Zstr
             +","+"Last_Souod_Or_Nozol_perc_Zstr"
             +","+this_day_CP[i]+","+Last_day_CP_Symbol[i]
             +","+symbol_des[i]+","+symbol_path[i];
string P14_FastCheck=numb_growD_5day[i]+","+numb_growD_10day[i]+","+numb_growD_20day[i]+","+numb_Posit_sig[i]
+","+numb_Nega_sig[i]+",Pos_Sigs :,"+P_Posit_sig+",Neg_Sigs :,"+P_Neg_Sig;
string P15__ASK_to_bid_Perc=ask_ta_bid_darsad;



//////// -------------------sell_Strategy_4 signals for day parameter
double Bar_AV_M0=(mrate_HH1[0].close+mrate_HH1[0].open)/2;
double Bar_AV_M1=(mrate_HH1[1].close+mrate_HH1[1].open)/2;
double Bar_AV_M2=(mrate_HH1[2].close+mrate_HH1[2].open)/2;
double Bar_AV_M3=(mrate_HH1[3].close+mrate_HH1[3].open)/2;
double Bar_AV_M4=(mrate_HH1[4].close+mrate_HH1[4].open)/2;
double shib_AV_M4_M3=Bar_AV_M3-Bar_AV_M4;
double shib_AV_M3_M2=Bar_AV_M2-Bar_AV_M3;
double shib_AV_M2_M1=Bar_AV_M1-Bar_AV_M2;
double shib_AV_M1_M0=Bar_AV_M0-Bar_AV_M1;



         if(!Sell_opened && Sell_opened
//		 (PTL_signal[0]=="Sell Signal_StartNozool" && PTL_Status[1]=="PTL_isNozooli" && PTL_Status[2]=="PTL_isNozooli" && PTL_Status[3]=="PTL_isNozooli") ||
//        (PTL_signal[0]=="Sell Signal_StartNozool" && PTL_Status[1]=="PTL_isNozooli" && PTL_Status[2]=="PTL_isNozooli") ||
//         (PTL_signal[0]=="Sell Signal_StartNozool" && PTL_Status[1]=="PTL_isNozooli" && PTL_Status[3]=="PTL_isNozooli") 

          //Ravand=="Ravand Nozooli" && ZigZag_Buffer[0]!=0   //&& shib_ravand[0]<=0.7*Ave_shib_rav_manfi
      // && ((Chg_RavandInPrice2nd_Zstr-last_price)/last_price)*100<=0.5*Ave__UptoDo_manf_Zst
         //SAR[1]<last_price && SAR[0]>last_price
         //shib_AV_M4_M3>0 && shib_AV_M3_M2>0 && shib_AV_M2_M1<0 && shib_AV_M1_M0<0 && shib_AV_M1_M0<shib_AV_M2_M1 //Bar_AV_M1>Bar_AV_M2 && mrate_HH1[0].open>Bar_AV_M1 && mrate_HH1[0].close>mrate_HH1[0].open
        // && CTi_Stru.hour<22

          //(CTi_Stru.hour==22 && CTi_Stru.min>29) //((mrate[0].close-mrate[0].open)/mrate[0].open*100)<(-1.5*Ave_100_Price_Perc[i])
          //((mrate[0].close-mrate[0].open)/mrate[0].open*100)<(-1.5*Ave_100_Price_Perc[i])
          //Ravand=="Ravand Nozooli" && Ravand_Zstr=="Ravand Nozooli" 
         //&& stochBuff_OSC[0]>35
         // && stochBuff_OSC[0]<signalBuff_OSC[0] && stochBuff_OSC[0]-stochBuff_OSC[1]<-2 && stochBuff_OSC[2]>stochBuff_OSC[1]
         //&& RSI[0]-RSI[1]<=-2 &&RSI[1]<RSI[2]
          //&& ((stochBuff_OSC[2]-stochBuff_OSC[1])<=(stochBuff_OSC[1]-stochBuff_OSC[0]))
          
         //stochBuff_OSC[0]>signalBuff_OSC[0] && stochBuff_OSC[0]-stochBuff_OSC[1]>2  //
           //&& RSI[0]-RSI[1]<=-2 
           //&&percent_Price_now[i]>-2
           //&&signalBuff_OSC[0]<50
           //&&stochBuff_OSC[0]>-20
           //&& RSI[1]<RSI[2]  //&& RSI[0] >50 && RSI[2]>RSI[3]
          //&& MACDBuffer[0]<MACDBuffer[1] //&& MACDSignal[0]>MACDSignal[1]  //&& signalBuff_OSC[0]>=stochBuff_OSC[0]
          //&&((deltaMACD1<deltaMACD2)) 
          //&& n_bar_Change_ravand[i]<=5
          //&& Strong_Signal_From_zigzag[i]==false 

          //&&((mrate_HH1[1].close-mrate_HH1[1].open)/mrate_HH1[1].open)*100<-0.1
          //&& ((mrate_HH1[2].close-mrate_HH1[2].open)/mrate_HH1[2].open)*100<-0.2
          //&& ((mrate_HH1[3].close-mrate_HH1[3].open)/mrate_HH1[3].open)*100<-0.2
          //Ravand=="Ravand Nozooli"  
          // && stochBuff_OSC[0]>35  && Ravand_Zstr=="Ravand Nozooli"
            )//     && && CTi_Stru.hour<22 (((last_price-Supp_Z)/Supp_Z)*100)>=-7           && mrate[0].tick_volume>(1.2*((mrate[1].tick_volume+ mrate[2].tick_volume)/2))   && mrate[0].tick_volume>(1.2*((mrate[1].tick_volume+ mrate[2].tick_volume)/2))  && (((last_price-Supp_Z)/Supp_Z)*100)<=7
           {
              
/*               if((Symbol_Last_sell_Time[i][1]==TimeToString(Ctime,TIME_DATE) && Symbol_Last_sell_Type[i][1]=="ZARAR"))
                {
                 continue; //adam kharid dor sorati ke emrooz kharid manfi dashtam
                } */
              
               tommorow_sell_dailyAlert[i]= true;

                Buy_Strategy_dailyAlert[i]="sell_Strategy_1";
                
          Alert("Strategy 1 sell_accept"+m_symbols_array[i],Email_Buy_Parameters[i]);
          tommorow_sell[i]= true;
          //Print("Time_date",TimeToString(mrate[0].time,TIME_DATE));
          date_of_accept_buy[i]=TimeToString(mrate[0].time,TIME_DATE);

             Sell_Strategy[i]=Buy_Strategy_dailyAlert[i];//"Buy_Strategy_1"
///---------
            P2_BuyStr=Buy_Strategy_dailyAlert[i]+","+tommorow_sell_dailyAlert[i];
            
            Last_Day_Symbol_parameters[i]=
            P1_symbol
            +","+P2_BuyStr
            +",Vazeiat bazar va shakhes,"
            +P3_Bazar_shakhes[i]
            //+","+P6_Win_B_or_S
            +","+P7_Indicators
            +","+P8_RavandP
            +","+P9_Sup_and_RessP
            +","+P10_just_dayP
            +","+P11_signals
            //+","+P12_candels
            //+",Extra param"
            //+","+P13_extraP
            //+",FastCheck"
            //+","+P14_FastCheck;
            +","+P15__ASK_to_bid_Perc
			   +","+MultiTF_Status;


             DayBef_Buy_parameters[i]=Last_Day_Symbol_parameters[i];
                
                
               }



//------------------------------------------- Write day Parameter File
//-------------------- make parameter daily----------------- 

P2_BuyStr=Buy_Strategy_dailyAlert[i]+","+tommorow_sell_dailyAlert[i];
      if(i<symbol_size)
        {
         Last_Day_Symbol_parameters[i]=
         P1_symbol
         +","+P2_BuyStr
         +",Vazeiat bazar va shakhes,"
         +P3_Bazar_shakhes[i]
         
         //+","+P6_Win_B_or_S
         +","+P7_Indicators
         +","+P8_RavandP
         +","+P9_Sup_and_RessP
         +","+P10_just_dayP
         
         +","+P11_signals
         //+","+P12_candels
         //+",Extra param"
         //+","+P13_extraP
         //+",FastCheck"
         //+","+P14_FastCheck;
         +","+P15__ASK_to_bid_Perc;
        }


         
        if(first_open_file1==false)
        {
         FileSeek(file_handle_2,0,SEEK_END);
         FileWrite(file_handle_2,Last_Day_Symbol_parameters[i]); 
         for(int j=0;j<size_Portfolio;j++)
           {
           if(m_symbols_array_Portfolio[j]==m_symbols_array[i])
             {
               FileSeek(file_handle_4,0,SEEK_END);
               FileWrite(file_handle_4,Last_Day_Symbol_parameters[i]); 
             }
           }
        }




//-----------------------------------End of sell Strategis -------------------------------//
   }//---it's End of if(!sell_opened )
   
   
   
 //----------------------------------START chack for CLOSE BUY-----------------------            
       //Print("Before IF We Have Open Buy Position  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
////-------------------------------- IF We Have Open buy Position, Check For close Time-----------------------------//// 

   if(m_Position.Select(m_symbols_array[i]))                     //if the position for this symbol already exists
      {
      //if(m_Position.PositionType()==POSITION_TYPE_SELL) continue;
      if(m_Position.PositionType()==POSITION_TYPE_BUY )   
         { 


        //----------------- define some parameter for close buy strategies---------------//

        //-Profit or loss of this Position
        double sood=(last_price*m_Position.Volume())-(m_Position.PriceOpen()*m_Position.Volume())
        -(m_Position.PriceOpen()*m_Position.Volume()*0.007)-(0.008*last_price*m_Position.Volume());           
        //-Time Parameters
         position_time=PositionGetInteger(POSITION_TIME);
        //datetime position_time=m_Position.Time();                            //Position Time
        string   position_date =TimeToString(m_Position.Time(),TIME_DATE);   //Psition Date String
        string   CTime_date=TimeToString(Ctime,TIME_DATE);                     //Current Date String



        int count_bar_FBuy=iBarShift(m_symbols_array[i],PERIOD_CURRENT, position_time); 
        int count_bar_FBuy_M1=iBarShift(m_symbols_array[i],PERIOD_M1, position_time);
        int count_bar_FBuy_H1=iBarShift(m_symbols_array[i],PERIOD_H1, position_time);
                 
 //---------------------------------------------------            
       //Print("Before save data after buy  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
//-------------------------save data after buy------------------------------ 
// save data after buy code 2--Start
      MqlRates mrate_H1_sell[];
      MqlRates mrate_min_sell[];
        


                  // H1 after buy  
                  ArraySetAsSeries(mrate_H1_sell,true);
                  int copied2_H1_sell=CopyRates(m_symbols_array[i],PERIOD_H1,position_time,Ctime,mrate_H1_sell); //Gets history data of MqlRates structure of a specified symbol-period in specified quantity into the rates_array array.
                  if(copied2_H1_sell>0)
                    { 
                          //Print("successful get history data for the symbol ,mrate_min ",m_symbols_array[i]); 
                    }
                    else
                    {
                     Print("Failed to get history data for the symbol ,copied2_H1_sell "
                     ,"  ","CTi_Stru.hour =",CTi_Stru.hour,"   ",m_symbols_array[i]); 
                     Alert("cant calc Last_day_CP_symbol",m_symbols_array[i]+"   Last CP change to 0 but no analysis in this symbol "
                     +"Failed to get history data for the symbol ,copied2_H1_sell "
                     );
                     //continue;
                    }
               // min after buy
                  ArraySetAsSeries(mrate_min_sell,true);
                  int copied2_sell=CopyRates(m_symbols_array[i],PERIOD_M1,position_time,Ctime,mrate_min_sell); //Gets history data of MqlRates structure of a specified symbol-period in specified quantity into the rates_array array.
                  if(copied2_sell>0)
                    { 
                          //Print("successful get history data for the symbol ,mrate_min ",m_symbols_array[i]); 
                    }
                    else
                    {
                     Print("Failed to get history data for the symbol ,mrate_min_sell "
                     ,"  ","CTi_Stru.hour =",CTi_Stru.hour,"   ",m_symbols_array[i]); 
                     Alert("cant calc Last_day_CP_symbol",m_symbols_array[i]+"   Last CP change to 0 but no analysis in this symbol "
                     +"Failed to get history data for the symbol ,mrate_min_sell "
                     );
                     continue;
                    }

               int k=0;
               double sumPV=0;
               double sum_vol=0;
               while(k<copied2_sell)//mrate_min[k].time>=StringToTime(TimeToString(Ctime,TIME_DATE)+" 9:00")
                 {
                 sum_vol=sum_vol+mrate_min_sell[k].tick_volume;
                  //Print("im in while",mrate_min[k].time,"      ",StringToTime(TimeToString(Ctime,TIME_DATE)+" 9:00" ));
                  //WVP[i]=WVP[i]+((((mrate_min[k].open))*mrate_min[k].tick_volume)/mrate[0].tick_volume);//mrate_min[k].high+mrate_min[k].low+mrate_min[k].close+
                  sumPV=sumPV+(((mrate_min_sell[k].high+mrate_min_sell[k].low+mrate_min_sell[k].close+mrate_min_sell[k].open)/4)*mrate_min_sell[k].tick_volume);//
                  //Print("WVP[i]",WVP[i],"k=",k,"bar time",mrate_min[k].time);
                  k++; 
                 }
                 
               double WVPBtoS=sumPV/sum_vol;
               //Test Last Price taghirat
               //double WVPBtoS=last_price;
               
               double percent_WVPBtoSell=((WVPBtoS-m_Position.PriceOpen())/m_Position.PriceOpen())*100;
               
               double SarBeSarPrice=m_Position.PriceOpen()*1.015;
                      
               if(copied2_sell>0 && copied2_sell<=60)
                 {
                  WVPB2S[i][0]=percent_WVPBtoSell;
                  VolB2S[i][0]=sum_vol;
                 }
               if(copied2_H1_sell==1)
                 {
                  WVPB2S[i][1]=percent_WVPBtoSell;
                  VolB2S[i][1]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==2)
                 {
                  WVPB2S[i][2]=percent_WVPBtoSell;
                  VolB2S[i][2]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==3)
                 {
                  WVPB2S[i][3]=percent_WVPBtoSell;
                  VolB2S[i][3]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==4)
                 {
                  WVPB2S[i][4]=percent_WVPBtoSell;
                  VolB2S[i][4]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==5)
                 {
                  WVPB2S[i][5]=percent_WVPBtoSell;
                  VolB2S[i][5]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==6)
                 {
                  WVPB2S[i][6]=percent_WVPBtoSell;
                  VolB2S[i][6]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==7)
                 {
                  WVPB2S[i][7]=percent_WVPBtoSell;
                  VolB2S[i][7]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==8)
                 {
                  WVPB2S[i][8]=percent_WVPBtoSell;
                  VolB2S[i][8]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==9)
                 {
                  WVPB2S[i][9]=percent_WVPBtoSell;
                  VolB2S[i][9]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==10)
                 {
                  WVPB2S[i][10]=percent_WVPBtoSell;
                  VolB2S[i][10]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==11)
                 {
                  WVPB2S[i][11]=percent_WVPBtoSell;
                  VolB2S[i][11]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==12)
                 {
                  WVPB2S[i][12]=percent_WVPBtoSell;
                  VolB2S[i][12]=mrate_H1_sell[0].tick_volume;
                 }
                 
                 
               if(copied2_H1_sell==13)
                 {
                  WVPB2S[i][13]=percent_WVPBtoSell;
                  VolB2S[i][13]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==14)
                 {
                  WVPB2S[i][14]=percent_WVPBtoSell;
                  VolB2S[i][14]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==15)
                 {
                  WVPB2S[i][15]=percent_WVPBtoSell;
                  VolB2S[i][15]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==16)
                 {
                  WVPB2S[i][16]=percent_WVPBtoSell;
                  VolB2S[i][16]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==17)
                 {
                  WVPB2S[i][17]=percent_WVPBtoSell;
                  VolB2S[i][17]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==18)
                 {
                  WVPB2S[i][18]=percent_WVPBtoSell;
                  VolB2S[i][18]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==19)
                 {
                  WVPB2S[i][19]=percent_WVPBtoSell;
                  VolB2S[i][19]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==20)
                 {
                  WVPB2S[i][20]=percent_WVPBtoSell;
                  VolB2S[i][20]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==21)
                 {
                  WVPB2S[i][21]=percent_WVPBtoSell;
                  VolB2S[i][21]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==22)
                 {
                  WVPB2S[i][22]=percent_WVPBtoSell;
                  VolB2S[i][22]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==23)
                 {
                  WVPB2S[i][23]=percent_WVPBtoSell;
                  VolB2S[i][23]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==24)
                 {
                  WVPB2S[i][24]=percent_WVPBtoSell;
                  VolB2S[i][24]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==25)
                 {
                  WVPB2S[i][25]=percent_WVPBtoSell;
                  VolB2S[i][25]=mrate_H1_sell[0].tick_volume;
                 }
// save data after buy code 2--END

//+------------------------------------------------------------------------------+
//|            -------------------All close buy Signals----------------------         |
//+------------------------------------------------------------------------------+
// Sell Signals code 2--Start

      double   profitt=m_Position.Profit(); //(((last_price-m_Position.PriceOpen())/m_Position.PriceOpen())*100)-1.5;
//----- max of sood and loos percent in this position
        
      if(profitt>MaxOfProfit_Position[i])
        {
         MaxOfProfit_Position[i]=profitt;
         H_MaxOfProfit_FromBuy[i]=count_bar_FBuy_H1;
        }
        
      if(profitt<MaxOfLoos_Position[i])
        {
         MaxOfLoos_Position[i]=profitt;
         H_MaxOfLoos_FromBuy[i]=count_bar_FBuy_H1;
         
//-----after 2 h shekast max of loos Warning
               if(copied2_H1_sell>2 && Email_count_Warning_afterBuy[i][3]<1)
                 {

         
                 Sell_signal_after_2_h_shekast_max_of_loos[i]=true;
                 Sell_signal_after_2_h_shekast_max_of_loos_Percent[i]=profitt;
                 Sell_signal_after_2_h_shekast_max_of_loos_Hour[i]=copied2_H1_sell;
                 Email_count_Warning_afterBuy[i][3]++;
                 }

        }



////-------------------------------------------------report sell signals Parameter
// Sell Signals code 2--Start
       string Sell_signals=SignalSellA[i]+","+SignalSellA_price[i]
        +","+SignalSellA_Loos[i]+","+SignalSellA_Hour[i]
        +","+SignalSellB[i]+","+SignalSellB_price[i]
        +","+SignalSellB_Loos[i]+","+SignalSellB_Hour[i]
        +","+SignalSellC[i]+","+SignalSellC_price[i]
        +","+SignalSellC_Loos[i]+","+SignalSellC_Hour[i]
        
        +","+shekast_Zigzag[i]+","+Pof_shekast_zigzag[i]
        +","+percent_shekast_zigzag[i]+","+shekast_Zigzag_Hour[i]
        +","+Sell_signal_Aft4H_MaxPBef3H[i]+","+Sell_signal_Aft4H_MaxPBef3H_Percent[i]
        +","+Sell_signal_Aft4H_MaxPBef3H_Hour[i]
        
        +","+Sell_signal_after_2_h_shekast_max_of_loos[i]
        +","+Sell_signal_after_2_h_shekast_max_of_loos_Percent[i]
        +","+Sell_signal_after_2_h_shekast_max_of_loos_Hour[i];
        
        
  //-----------sell Parameter
        string Sell_Parameter=count_bar_FBuy
        +","+MaxOfProfit_Position[i]+","+H_MaxOfProfit_FromBuy[i]
        +","+MaxOfLoos_Position[i]+","+H_MaxOfLoos_FromBuy[i];

      //Print(m_symbols_array[i],"numb bar from Buy",count_bar_FBuy)   ;  
string Sell_time_Write_data=Ctime+","+m_symbols_array[i]+","+profitt+","+m_Position.Time()+","+m_Position.Comment()
+",After_Buy,"+Sell_Parameter+","+Sell_signals+","+DayBef_Buy_parameters[i];

string Sell_signals_S="SignalSellA[i],SignalSellA_price[i],SignalSellA_Loos[i],SignalSellA_Hour[i],SignalSellB[i],SignalSellB_price[i]"
      +",SignalSellB_Loos[i],SignalSellB_Hour[i],SignalSellC[i],SignalSellC_price[i],SignalSellC_Loos[i],SignalSellC_Hour[i]"
      +",shekast_Zigzag[i],Pof_shekast_zigzag[i],percent_shekast_zigzag[i],shekast_Zigzag_Hour[i],Sell_signal_Aft4H_MaxPBef3H[i]"
      +",Sell_signal_Aft4H_MaxPBef3H_Percent[i],Sell_signal_Aft4H_MaxPBef3H_Hour[i],Sell_signal_after_2_h_shekast_max_of_loos[i]"
      +",Sell_signal_after_2_h_shekast_max_of_loos_Percent[i],Sell_signal_after_2_h_shekast_max_of_loos_Hour[i]";
      
string Sell_Parameter_S="numb_D_buy_to_sell,MaxOfProfit_Position[i],H_MaxOfProfit_FromBuy[i],MaxOfLoos_Position[i],H_MaxOfLoos_FromBuy[i]";

string Excel_first_row=
      "P type,P Strategy,close P Time,symbol,Sood_zarar,Open P Time"
      +",After_Buy"//After_Buy
      +","+Sell_Parameter_S
      +","+Sell_signals_S
      +","+day_parameter_first_row+","+MultiTF_Status_Header+"\r\n";




//----------------- Python input file created and RUN  ----------


      if (PythonCreatedFileTime_BuyCheck!=Ctime && FirstCreatedPythonFile_BuyCheck==true)
      {
      count_M1_PythonFileCreated_BuyCheck=iBarShift(m_symbols_array[i],PERIOD_M1, PythonCreatedFileTime_BuyCheck);
      }
      
            if (FirstCreatedPythonFile_BuyCheck == false || count_M1_PythonFileCreated_BuyCheck > 1)
              {
               // حذف فایل قبلی از پوشه مشترک
               FileDelete("inputFile_Python.csv", FILE_COMMON);
            
               // ایجاد فایل جدید در پوشه مشترک
               int file_handle_inputPython = FileOpen("inputFile_Python.csv", FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON);
               if(file_handle_inputPython == INVALID_HANDLE)
                 {
                  PrintFormat("❌ Failed to open %s file, Error code = %d", "inputFile_Python.csv", GetLastError());
                 }
               else
                 {
                  FileWriteString(file_handle_inputPython, MultiTF_Status_Header + "\r\n" + MultiTF_Status);
                  FileClose(file_handle_inputPython);
                  Print("✅ inputFile_Python.csv created in Common\\Files");
                 }
            
               PythonCreatedFileTime_BuyCheck = Ctime;
               FirstCreatedPythonFile_BuyCheck = true;
               count_M1_PythonFileCreated_BuyCheck = 0;
            
               // کپی فایل در همان پوشه مشترک
               FileCopy("inputFile_Python.csv", FILE_COMMON, "inputFile_Python2.csv", FILE_COMMON);
              }

        
             // 3. انتظار برای تولید خروجی
             Sleep(20000);  // 2 ثانیه صبر کن (می‌تونی بهبود بدی با چک کردن وجود فایل)
             //    FileDelete("inputFile_Python.csv");
             
             // 4. خواندن خروجی
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
            double p_15min = loader.GetProfitProbByTag("15minBuy");
            double n_15min = loader.GetNeutralProbByTag("15minBuy");
            double l_15min = loader.GetLossProbByTag("15minBuy");
            
            double p_1h = loader.GetProfitProbByTag("1hBuy");
            double n_1h = loader.GetNeutralProbByTag("1hBuy");
            double l_1h = loader.GetLossProbByTag("1hBuy");
         
            double p_2h = loader.GetProfitProbByTag("2hBuy");
            double n_2h = loader.GetNeutralProbByTag("2hBuy");
            double l_2h = loader.GetLossProbByTag("2hBuy");
         
            double p_3h = loader.GetProfitProbByTag("3hBuy");
            double n_3h = loader.GetNeutralProbByTag("3hBuy");
            double l_3h = loader.GetLossProbByTag("3hBuy");
            
            double p_4h = loader.GetProfitProbByTag("4hBuy");
            double n_4h = loader.GetNeutralProbByTag("4hBuy");
            double l_4h = loader.GetLossProbByTag("4hBuy");
            
            double p_1d = loader.GetProfitProbByTag("1DBuy");
            double n_1d = loader.GetNeutralProbByTag("1DBuy");
            double l_1d = loader.GetLossProbByTag("1DBuy");
            
            FileDelete("prediction_15minBuy.txt", FILE_COMMON);
            FileDelete("prediction_1hBuy.txt", FILE_COMMON);
            FileDelete("prediction_2hBuy.txt", FILE_COMMON);
            FileDelete("prediction_3hBuy.txt", FILE_COMMON);
            FileDelete("prediction_4hBuy.txt", FILE_COMMON);
            FileDelete("prediction_1DBuy.txt", FILE_COMMON);

//---------------------------------------
       //Print("Before  tatmam halat haye sell  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]); 
              /////////////  tatmam halat haye sell  //////////////
/////+++++++++++/////////// check kardan Hade zarar 3 (close buy) ///////+++++++++++//////////
double sood_zara_USD=m_Position.Profit(); //(last_price-m_Position.PriceOpen())*m_Position.Volume()*100;


double Bar_AV_M0=(mrate_HH1[0].close+mrate_HH1[0].open)/2;
double Bar_AV_M1=(mrate_HH1[1].close+mrate_HH1[1].open)/2;
double Bar_AV_M2=(mrate_HH1[2].close+mrate_HH1[2].open)/2;
double Bar_AV_M3=(mrate_HH1[3].close+mrate_HH1[3].open)/2;
double Bar_AV_M4=(mrate_HH1[4].close+mrate_HH1[4].open)/2;
double shib_AV_M4_M3=Bar_AV_M3-Bar_AV_M4;
double shib_AV_M3_M2=Bar_AV_M2-Bar_AV_M3;
double shib_AV_M2_M1=Bar_AV_M1-Bar_AV_M2;
double shib_AV_M1_M0=Bar_AV_M0-Bar_AV_M1;


      if( copied2_H1_sell>=24 || (l_4h>0.8 && l_3h>0.8 && (l_2h>0.8 || l_1h>0.8) && copied2_H1_sell<20 )
	  //PTL_signal[0] ==  "Sell Signal_StartNozool" || PTL_Status[0] == "PTL_isNeutral" || PTL_Status[0] == "PTL_isNozooli"
      //sood_zara_USD<=-5 //|| ((last_price-m_Position.PriceOpen())/m_Position.PriceOpen())*100<-0.5*Ave__UptoDo_mosb
       //((last_price-m_Position.PriceOpen())/m_Position.PriceOpen())*100<-1.5 // && 
       )
              { 
              
            if(m_Trade.PositionClose(m_Position.Ticket()))		// haminja baste mishe
              {
              
               
         if(first_open_file2==true)
         {
         FileWrite(file_handle,Excel_first_row
         +"PTL_Close"+","+m_Position.Comment()+","+Ctime+","+m_symbols_array[i]
         +","+sood_zara_USD+","+m_Position.Time()+",After_Buy,"+Sell_Parameter+","+Sell_signals+","+DayBef_Buy_parameters[i]);  
         first_open_file2=false;
         }
         else
         {
         FileSeek(file_handle,0,SEEK_END);
         FileWrite(file_handle,
         "PTL_Close"+","+m_Position.Comment()+","+Ctime+","+m_symbols_array[i]
         +","+sood_zara_USD+","+m_Position.Time()+",After_Buy,"+Sell_Parameter+","+Sell_signals+","+DayBef_Buy_parameters[i]);   
          }
               

         mrequest.comment="PTL_Close" ;

         Symbol_Last_Buy_Time[i][1]=TimeToString(Ctime,TIME_DATE);  
         Symbol_Last_Buy_Type[i][1]="ZARAR";
         
     SignalSellA[i]=false;
     SignalSellA_price[i]=0;
     SignalSellA_Loos[i]=0;
     SignalSellB[i]=false;
     SignalSellB_price[i]=0;
     SignalSellB_Loos[i]=0;
     SignalSellC[i]=false;
     SignalSellC_price[i]=0;
     SignalSellC_Loos[i]=0;
     
     Pof_shekast_zigzag[i]=0;
     shekast_Zigzag[i]=false;
     Sell_signal_Aft4H_MaxPBef3H[i]=false;
     Sell_signal_after_2_h_shekast_max_of_loos[i]=false;

     Sell_signal_after_2_h_shekast_max_of_loos_Percent[i]=0;
     percent_shekast_zigzag[i]=0;
     Sell_signal_Aft4H_MaxPBef3H_Percent[i]=0;
     Sell_signal_after_2_h_shekast_max_of_loos_Hour[i]=0;
     SignalSellA_Hour[i]=0;
     SignalSellB_Hour[i]=0;
     SignalSellC_Hour[i]=0;
     Sell_signal_Aft4H_MaxPBef3H_Hour[i]=0;
     shekast_Zigzag_Hour[i]=0;

     MaxOfProfit_Position[i]=0;
     H_MaxOfProfit_FromBuy[i]=0;
     MaxOfLoos_Position[i]=0;
     H_MaxOfLoos_FromBuy[i]=0;
     // set zero Array

        for(int j=0;j<26;j++)
          {
           WVPB2S[i][j]=0;
           VolB2S[i][j]=0;
          }
          
     // set zero Array
        for(int j=0;j<10;j++)
          {
           WVPBef2B[i][j]=0;
          }
        // set zero Array
        for(int j=0;j<27;j++)
          {
           Email_count_Warning_afterBuy[i][j]=0;
          }
          
          
         }


              }   
              
		//--------------------------------------------------------------------------
		//--------------------------------------------------------------------------
         }//end of close buy 1
       }//end of close buy 2
 //-----------------------------------End Of Check for Close BUY 


 //-----------------------------------Start Check for Close SELL 
       //Print("Before IF We Have Open Buy Position  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
////-------------------------------- IF We Have Open sell Position, Check For close Time-----------------------------//// 

   if(m_Position.Select(m_symbols_array[i]))                     //if the position for this symbol already exists
      {
      //if(m_Position.PositionType()==POSITION_TYPE_SELL) continue;
      if(m_Position.PositionType()==POSITION_TYPE_SELL )   
         { 


        //----------------- define some parameter for close sell strategies---------------//

        //-Profit or loss of this Position
        double sood=m_Position.Profit(); //(last_price*m_Position.Volume())-(m_Position.PriceOpen()*m_Position.Volume())
        //-(m_Position.PriceOpen()*m_Position.Volume()*0.007)-(0.008*last_price*m_Position.Volume());           
        //-Time Parameters
        
         position_time=m_Position.Time();                            //Position Time
        string   position_date =TimeToString(m_Position.Time(),TIME_DATE);   //Psition Date String
        string   CTime_date=TimeToString(Ctime,TIME_DATE);                     //Current Date String



        int count_bar_FBuy=iBarShift(m_symbols_array[i],PERIOD_CURRENT, position_time); 
        int count_bar_FBuy_M1=iBarShift(m_symbols_array[i],PERIOD_M1, position_time);
        int count_bar_FBuy_H1=iBarShift(m_symbols_array[i],PERIOD_H1, position_time);
                 
 //---------------------------------------------------            
       //Print("Before save data after buy  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]);  
//-------------------------save data after buy------------------------------ 
// save data after buy code 2--Start
      MqlRates mrate_H1_sell[];
      MqlRates mrate_min_sell[];
        


                  // H1 after buy  
                  ArraySetAsSeries(mrate_H1_sell,true);
                  int copied2_H1_sell=CopyRates(m_symbols_array[i],PERIOD_H1,position_time,Ctime,mrate_H1_sell); //Gets history data of MqlRates structure of a specified symbol-period in specified quantity into the rates_array array.
                  if(copied2_H1_sell>0)
                    { 
                          //Print("successful get history data for the symbol ,mrate_min ",m_symbols_array[i]); 
                    }
                    else
                    {
                     Print("Failed to get history data for the symbol ,copied2_H1_sell "
                     ,"  ","CTi_Stru.hour =",CTi_Stru.hour,"   ",m_symbols_array[i]); 
                     Alert("cant calc Last_day_CP_symbol",m_symbols_array[i]+"   Last CP change to 0 but no analysis in this symbol "
                     +"Failed to get history data for the symbol ,copied2_H1_sell "
                     );
                     //continue;
                    }
               // min after buy
                  ArraySetAsSeries(mrate_min_sell,true);
                  int copied2_sell=CopyRates(m_symbols_array[i],PERIOD_M1,position_time,Ctime,mrate_min_sell); //Gets history data of MqlRates structure of a specified symbol-period in specified quantity into the rates_array array.
                  if(copied2_sell>0)
                    { 
                          //Print("successful get history data for the symbol ,mrate_min ",m_symbols_array[i]); 
                    }
                    else
                    {
                     Print("Failed to get history data for the symbol ,mrate_min_sell "
                     ,"  ","CTi_Stru.hour =",CTi_Stru.hour,"   ",m_symbols_array[i]); 
                     Alert("cant calc Last_day_CP_symbol",m_symbols_array[i]+"   Last CP change to 0 but no analysis in this symbol "
                     +"Failed to get history data for the symbol ,mrate_min_sell "
                     );
                     continue;
                    }

               int k=0;
               double sumPV=0;
               double sum_vol=0;
               while(k<copied2_sell)//mrate_min[k].time>=StringToTime(TimeToString(Ctime,TIME_DATE)+" 9:00")
                 {
                 sum_vol=sum_vol+mrate_min_sell[k].tick_volume;
                  //Print("im in while",mrate_min[k].time,"      ",StringToTime(TimeToString(Ctime,TIME_DATE)+" 9:00" ));
                  //WVP[i]=WVP[i]+((((mrate_min[k].open))*mrate_min[k].tick_volume)/mrate[0].tick_volume);//mrate_min[k].high+mrate_min[k].low+mrate_min[k].close+
                  sumPV=sumPV+(((mrate_min_sell[k].high+mrate_min_sell[k].low+mrate_min_sell[k].close+mrate_min_sell[k].open)/4)*mrate_min_sell[k].tick_volume);//
                  //Print("WVP[i]",WVP[i],"k=",k,"bar time",mrate_min[k].time);
                  k++; 
                 }
                 
               double WVPBtoS=sumPV/sum_vol;
               //Test Last Price taghirat
               //double WVPBtoS=last_price;
               
               double percent_WVPBtoSell=((WVPBtoS-m_Position.PriceOpen())/m_Position.PriceOpen())*100;
               
               double SarBeSarPrice=m_Position.PriceOpen()*1.015;
                      
               if(copied2_sell>0 && copied2_sell<=60)
                 {
                  WVPB2S[i][0]=percent_WVPBtoSell;
                  VolB2S[i][0]=sum_vol;
                 }
               if(copied2_H1_sell==1)
                 {
                  WVPB2S[i][1]=percent_WVPBtoSell;
                  VolB2S[i][1]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==2)
                 {
                  WVPB2S[i][2]=percent_WVPBtoSell;
                  VolB2S[i][2]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==3)
                 {
                  WVPB2S[i][3]=percent_WVPBtoSell;
                  VolB2S[i][3]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==4)
                 {
                  WVPB2S[i][4]=percent_WVPBtoSell;
                  VolB2S[i][4]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==5)
                 {
                  WVPB2S[i][5]=percent_WVPBtoSell;
                  VolB2S[i][5]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==6)
                 {
                  WVPB2S[i][6]=percent_WVPBtoSell;
                  VolB2S[i][6]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==7)
                 {
                  WVPB2S[i][7]=percent_WVPBtoSell;
                  VolB2S[i][7]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==8)
                 {
                  WVPB2S[i][8]=percent_WVPBtoSell;
                  VolB2S[i][8]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==9)
                 {
                  WVPB2S[i][9]=percent_WVPBtoSell;
                  VolB2S[i][9]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==10)
                 {
                  WVPB2S[i][10]=percent_WVPBtoSell;
                  VolB2S[i][10]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==11)
                 {
                  WVPB2S[i][11]=percent_WVPBtoSell;
                  VolB2S[i][11]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==12)
                 {
                  WVPB2S[i][12]=percent_WVPBtoSell;
                  VolB2S[i][12]=mrate_H1_sell[0].tick_volume;
                 }
                 
                 
               if(copied2_H1_sell==13)
                 {
                  WVPB2S[i][13]=percent_WVPBtoSell;
                  VolB2S[i][13]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==14)
                 {
                  WVPB2S[i][14]=percent_WVPBtoSell;
                  VolB2S[i][14]=mrate_H1_sell[0].tick_volume;
                 }
                 
               if(copied2_H1_sell==15)
                 {
                  WVPB2S[i][15]=percent_WVPBtoSell;
                  VolB2S[i][15]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==16)
                 {
                  WVPB2S[i][16]=percent_WVPBtoSell;
                  VolB2S[i][16]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==17)
                 {
                  WVPB2S[i][17]=percent_WVPBtoSell;
                  VolB2S[i][17]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==18)
                 {
                  WVPB2S[i][18]=percent_WVPBtoSell;
                  VolB2S[i][18]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==19)
                 {
                  WVPB2S[i][19]=percent_WVPBtoSell;
                  VolB2S[i][19]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==20)
                 {
                  WVPB2S[i][20]=percent_WVPBtoSell;
                  VolB2S[i][20]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==21)
                 {
                  WVPB2S[i][21]=percent_WVPBtoSell;
                  VolB2S[i][21]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==22)
                 {
                  WVPB2S[i][22]=percent_WVPBtoSell;
                  VolB2S[i][22]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==23)
                 {
                  WVPB2S[i][23]=percent_WVPBtoSell;
                  VolB2S[i][23]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==24)
                 {
                  WVPB2S[i][24]=percent_WVPBtoSell;
                  VolB2S[i][24]=mrate_H1_sell[0].tick_volume;
                 }
               if(copied2_H1_sell==25)
                 {
                  WVPB2S[i][25]=percent_WVPBtoSell;
                  VolB2S[i][25]=mrate_H1_sell[0].tick_volume;
                 }
// save data after buy code 2--END
//+------------------------------------------------------------------------------+
//|            -------------------All close Sell Signals----------------------         |
//+------------------------------------------------------------------------------+
// close Sell Signals code 2--Start

      double   profitt=m_Position.Profit();//(((last_price-m_Position.PriceOpen())/m_Position.PriceOpen())*100)-1.5;
//----- max of sood and loos percent in this position
        
      if(profitt>MaxOfProfit_Position[i])
        {
         MaxOfProfit_Position[i]=profitt;
         H_MaxOfProfit_FromBuy[i]=count_bar_FBuy_H1;
        }
        
      if(profitt<MaxOfLoos_Position[i])
        {
         MaxOfLoos_Position[i]=profitt;
         H_MaxOfLoos_FromBuy[i]=count_bar_FBuy_H1;
         
//-----after 2 h shekast max of loos Warning
               if(copied2_H1_sell>2 && Email_count_Warning_afterBuy[i][3]<1)
                 {

         
                 Sell_signal_after_2_h_shekast_max_of_loos[i]=true;
                 Sell_signal_after_2_h_shekast_max_of_loos_Percent[i]=profitt;
                 Sell_signal_after_2_h_shekast_max_of_loos_Hour[i]=copied2_H1_sell;
                 Email_count_Warning_afterBuy[i][3]++;
                 }

        }
// Sell Signals code 2--END


////----report sell signals Parameter
// Sell Signals code 2--Start
       string Sell_signals=SignalSellA[i]+","+SignalSellA_price[i]
        +","+SignalSellA_Loos[i]+","+SignalSellA_Hour[i]
        +","+SignalSellB[i]+","+SignalSellB_price[i]
        +","+SignalSellB_Loos[i]+","+SignalSellB_Hour[i]
        +","+SignalSellC[i]+","+SignalSellC_price[i]
        +","+SignalSellC_Loos[i]+","+SignalSellC_Hour[i]
        
        +","+shekast_Zigzag[i]+","+Pof_shekast_zigzag[i]
        +","+percent_shekast_zigzag[i]+","+shekast_Zigzag_Hour[i]
        +","+Sell_signal_Aft4H_MaxPBef3H[i]+","+Sell_signal_Aft4H_MaxPBef3H_Percent[i]
        +","+Sell_signal_Aft4H_MaxPBef3H_Hour[i]
        
        +","+Sell_signal_after_2_h_shekast_max_of_loos[i]
        +","+Sell_signal_after_2_h_shekast_max_of_loos_Percent[i]
        +","+Sell_signal_after_2_h_shekast_max_of_loos_Hour[i];
        
        
  //-----------sell Parameter
string Sell_Parameter=count_bar_FBuy
        +","+MaxOfProfit_Position[i]+","+H_MaxOfProfit_FromBuy[i]
        +","+MaxOfLoos_Position[i]+","+H_MaxOfLoos_FromBuy[i];

      //Print(m_symbols_array[i],"numb bar from Buy",count_bar_FBuy)   ;  
string Sell_time_Write_data=Ctime+","+m_symbols_array[i]+","+profitt+","+m_Position.Time()+","+m_Position.Comment()
+",After_Buy,"+Sell_Parameter+","+Sell_signals+","+DayBef_Buy_parameters[i];

string Sell_signals_S="SignalSellA[i],SignalSellA_price[i],SignalSellA_Loos[i],SignalSellA_Hour[i],SignalSellB[i],SignalSellB_price[i]"
      +",SignalSellB_Loos[i],SignalSellB_Hour[i],SignalSellC[i],SignalSellC_price[i],SignalSellC_Loos[i],SignalSellC_Hour[i]"
      +",shekast_Zigzag[i],Pof_shekast_zigzag[i],percent_shekast_zigzag[i],shekast_Zigzag_Hour[i],Sell_signal_Aft4H_MaxPBef3H[i]"
      +",Sell_signal_Aft4H_MaxPBef3H_Percent[i],Sell_signal_Aft4H_MaxPBef3H_Hour[i],Sell_signal_after_2_h_shekast_max_of_loos[i]"
      +",Sell_signal_after_2_h_shekast_max_of_loos_Percent[i],Sell_signal_after_2_h_shekast_max_of_loos_Hour[i]";
      
string Sell_Parameter_S="numb_D_buy_to_sell,MaxOfProfit_Position[i],H_MaxOfProfit_FromBuy[i],MaxOfLoos_Position[i],H_MaxOfLoos_FromBuy[i]";

string Excel_first_row=
      "P type,P Strategy,close P Time,symbol,Sood_zarar,Open P Time"
      +",After_Buy"//After_Buy
      +","+Sell_Parameter_S
      +","+Sell_signals_S
      +","+day_parameter_first_row+","+MultiTF_Status_Header+"\r\n";

     
//---------------------------------------
       //Print("Before  tatmam halat haye sell  ",m_symbols_array[i],
       //" tick_numb ",tick_numb," Time ",Ctime," counter_array[i] ",counter_array[i]); 
              /////////////  tatmam halat haye sell  //////////////
/////+++++++++++/////////// check kardan Hade zarar 5 (close sell) ///////+++++++++++//////////
double sood_zara_USD=m_Position.Profit(); //(-last_price+m_Position.PriceOpen())*m_Position.Volume()*100;

double Bar_AV_M0=(mrate_HH1[0].close+mrate_HH1[0].open)/2;
double Bar_AV_M1=(mrate_HH1[1].close+mrate_HH1[1].open)/2;
double Bar_AV_M2=(mrate_HH1[2].close+mrate_HH1[2].open)/2;
double Bar_AV_M3=(mrate_HH1[3].close+mrate_HH1[3].open)/2;
double Bar_AV_M4=(mrate_HH1[4].close+mrate_HH1[4].open)/2;
double shib_AV_M4_M3=Bar_AV_M3-Bar_AV_M4;
double shib_AV_M3_M2=Bar_AV_M2-Bar_AV_M3;
double shib_AV_M2_M1=Bar_AV_M1-Bar_AV_M2;
double shib_AV_M1_M0=Bar_AV_M0-Bar_AV_M1;


      if( copied2_H1_sell>=2
	  //PTL_signal[0] ==  "Buy Signa_StartSouod" || PTL_Status[0] == "PTL_isNeutral" || PTL_Status[0] == "PTL_isSouodi"
       //((last_price-m_Position.PriceOpen())/m_Position.PriceOpen())*100>1.5 // && 
       )
              { 
              
            if(m_Trade.PositionClose(m_Position.Ticket()))
              {
              
         if(first_open_file2==true)
         {
         FileWrite(file_handle,Excel_first_row
         +"PTL_Close"+","+m_Position.Comment()+","+Ctime+","+m_symbols_array[i]
         +","+sood_zara_USD+","+m_Position.Time()+",After_Buy,"+Sell_Parameter+","+Sell_signals+","+DayBef_Buy_parameters[i]);  
         first_open_file2=false;
         }
         else
         {
         FileSeek(file_handle,0,SEEK_END);
         FileWrite(file_handle,
         "PTL_Close"+","+m_Position.Comment()+","+Ctime+","+m_symbols_array[i]
         +","+sood_zara_USD+","+m_Position.Time()+",After_Buy,"+Sell_Parameter+","+Sell_signals+","+DayBef_Buy_parameters[i]);   
          }
               
               
         mrequest.comment="PTL_Close" ;

         Symbol_Last_sell_Time[i][1]=TimeToString(Ctime,TIME_DATE);  
         Symbol_Last_sell_Type[i][1]="ZARAR";
         
     SignalSellA[i]=false;
     SignalSellA_price[i]=0;
     SignalSellA_Loos[i]=0;
     SignalSellB[i]=false;
     SignalSellB_price[i]=0;
     SignalSellB_Loos[i]=0;
     SignalSellC[i]=false;
     SignalSellC_price[i]=0;
     SignalSellC_Loos[i]=0;
     
     Pof_shekast_zigzag[i]=0;
     shekast_Zigzag[i]=false;
     Sell_signal_Aft4H_MaxPBef3H[i]=false;
     Sell_signal_after_2_h_shekast_max_of_loos[i]=false;

     Sell_signal_after_2_h_shekast_max_of_loos_Percent[i]=0;
     percent_shekast_zigzag[i]=0;
     Sell_signal_Aft4H_MaxPBef3H_Percent[i]=0;
     Sell_signal_after_2_h_shekast_max_of_loos_Hour[i]=0;
     SignalSellA_Hour[i]=0;
     SignalSellB_Hour[i]=0;
     SignalSellC_Hour[i]=0;
     Sell_signal_Aft4H_MaxPBef3H_Hour[i]=0;
     shekast_Zigzag_Hour[i]=0;

     MaxOfProfit_Position[i]=0;
     H_MaxOfProfit_FromBuy[i]=0;
     MaxOfLoos_Position[i]=0;
     H_MaxOfLoos_FromBuy[i]=0;
     // set zero Array

        for(int j=0;j<26;j++)
          {
           WVPB2S[i][j]=0;
           VolB2S[i][j]=0;
          }
          
     // set zero Array
        for(int j=0;j<10;j++)
          {
           WVPBef2B[i][j]=0;
          }
        // set zero Array
        for(int j=0;j<27;j++)
          {
           Email_count_Warning_afterBuy[i][j]=0;
          }
          
          
            }

              }   
 


///---------------------
              
         }//end of close sell 1
       }//end of close sell 2
	   
//-----------------------------------End Of Check for Close SELL 
         
//---------------------------------------
         
//15day parameter code 2--Start
if(i<symbol_size && perc_Price_15day[i]>6)
  {
   Perc20d_upper6perc_cunt=Perc20d_upper6perc_cunt+1;
   //Perc20d_upper6perc_cunt_sanat[Sanaat[i]]=Perc20d_upper6perc_cunt_sanat[Sanaat[i]]+1;
  }
if(i<symbol_size && perc_Price_15day[i]>10)
  {
   Perc20d_upper10perc_cunt=Perc20d_upper10perc_cunt+1;
   //Perc20d_upper10perc_cunt_sanat[Sanaat[i]]=Perc20d_upper10perc_cunt_sanat[Sanaat[i]]+1;
  }
if(i<symbol_size && perc_Price_15day[i]>20)
  {
   Perc20d_upper20perc_cunt=Perc20d_upper20perc_cunt+1;
   //Perc20d_upper20perc_cunt_sanat[Sanaat[i]]=Perc20d_upper20perc_cunt_sanat[Sanaat[i]]+1;
  }
if(i<symbol_size && perc_Price_15day[i]>30)
  {
   Perc20d_upper30perc_cunt=Perc20d_upper30perc_cunt+1;
   //Perc20d_upper30perc_cunt_sanat[Sanaat[i]]=Perc20d_upper30perc_cunt_sanat[Sanaat[i]]+1;
  }
if(i<symbol_size && perc_Price_15day[i]<6)
  {
   Perc20d_lower6perc_cunt=Perc20d_lower6perc_cunt+1;
   //Perc20d_lower6perc_cunt_sanat[Sanaat[i]]=Perc20d_lower6perc_cunt_sanat[Sanaat[i]]+1;
  }
  
if(i<symbol_size && perc_Price_15day[i]<-6)
  {
   Perc20d_lowerneg6perc_cunt=Perc20d_lowerneg6perc_cunt+1;
   //Perc20d_lowerneg6perc_cunt_sanat[Sanaat[i]]=Perc20d_lowerneg6perc_cunt_sanat[Sanaat[i]]+1;
  }
if(i<symbol_size && perc_Price_15day[i]<-10)
  {
   Perc20d_lowerneg10perc_cunt=Perc20d_lowerneg10perc_cunt+1;
   //Perc20d_lowerneg10perc_cunt_sanat[Sanaat[i]]=Perc20d_lowerneg10perc_cunt_sanat[Sanaat[i]]+1;
  }
if(i<symbol_size && perc_Price_15day[i]<-20)
  {
   Perc20d_lowerneg20perc_cunt=Perc20d_lowerneg20perc_cunt+1;
   //Perc20d_lowerneg20perc_cunt_sanat[Sanaat[i]]=Perc20d_lowerneg20perc_cunt_sanat[Sanaat[i]]+1;
  }
//15day parameter code 2--END
//-------
if(i<symbol_size && percent_Price_now[i]>0)
  {
   Posit_symb_cunt=Posit_symb_cunt+1;
  }
if(i<symbol_size && percent_Price_now[i]<0)
  {
   Nega_symb_cunt=Nega_symb_cunt+1;
  }
if(i<symbol_size && percent_Price_now[i]==0)
  {
   Zero_symb_cunt=Zero_symb_cunt+1;
  }
if(i<symbol_size && percent_Price_now[i]>=3)
  {
   Up_mos3_symb_cunt=Up_mos3_symb_cunt+1;
  }
if(i<symbol_size && percent_Price_now[i]<=-3)
  {
   Low_manf3_symb_cunt=Low_manf3_symb_cunt+1;
  }
  

      
     }// end of for
     
     
// 15day analysis code 2--Start

if(checked_symbol_count!=0)
  {
   avr_numb_Nega_sig=sum_numb_Nega_sig/checked_symbol_count;
   avr_numb_Posit_sig=sum_numb_Posit_sig/checked_symbol_count;
  }


      if(first_open_file3==true)
      {

       FileWrite(file_handle_3,
             "Today_TIME"
             +",After15dayparam"
             +",Perc20d_upper30perc_cunt"
             +",Perc20d_upper20perc_cunt"
             +",Perc20d_upper10perc_cunt"
             +",Perc20d_upper6perc_cunt"
             +",Perc20d_lower6perc_cunt"
             +",Perc20d_lowerneg6perc_cunt"
             +",Perc20d_lowerneg10perc_cunt"
             +",Perc20d_lowerneg20perc_cunt"
             +",dayparam"
             +",shakhes_perc"
             +",shakhes_Hamvazn_perc"
             +",total_cost"
             +",Posit_symb_cunt"
             +",Nega_symb_cunt"
             +",Zero_symb_cunt"
             +",Up_mos3_symb_cunt"
             +",Low_manf3_symb_cunt"
             +",shakhes_to_Sup_perc"
             +",shakhes_to_Res_perc"
             +",avr_numb_Posit_sig"
             +",avr_numb_Nega_sig\r\n"
             
             +Day_Analysis_for_15day);

         first_open_file3=false;
        }
if(CTi_Stru.hour>=12 && CTi_Stru.min>=20 && CTi_Stru.hour<13 && CTi_Stru.min<=40
   && (TimeToString(Ctime,TIME_DATE)!=LastTimeWritefile3 || LastTimeWritefile3=="none" ) )
  {
             Day_Analysis_for_15day=
             Ctime
             +",After15dayparam"
             +","+Perc20d_upper30perc_cunt
             +","+Perc20d_upper20perc_cunt
             +","+Perc20d_upper10perc_cunt
             +","+Perc20d_upper6perc_cunt
             +","+Perc20d_lower6perc_cunt
             +","+Perc20d_lowerneg6perc_cunt
             +","+Perc20d_lowerneg10perc_cunt
             +","+Perc20d_lowerneg20perc_cunt
             +",dayparam,"
             +shakhes_perc
             +","+shakhes_Hamvazn_perc
             +","+total_cost
             +","+Posit_symb_cunt
             +","+Nega_symb_cunt
             +","+Zero_symb_cunt
             +","+Up_mos3_symb_cunt
             +","+Low_manf3_symb_cunt
             +","+shakhes_to_Sup_perc
             +","+shakhes_to_Res_perc
             +","+avr_numb_Posit_sig
             +","+avr_numb_Nega_sig;
             
        if(first_open_file3==false)
        {
         FileSeek(file_handle_3,0,SEEK_END);
         FileWrite(file_handle_3,Day_Analysis_for_15day); 
         LastTimeWritefile3=TimeToString(Ctime,TIME_DATE);
        }
        
  }


   Perc20d_upper30perc_cunt=0;
   Perc20d_upper20perc_cunt=0;
   Perc20d_upper10perc_cunt=0;
   Perc20d_upper6perc_cunt=0;
   Perc20d_lower6perc_cunt=0;
   Perc20d_lowerneg6perc_cunt=0;
   Perc20d_lowerneg10perc_cunt=0;
   Perc20d_lowerneg20perc_cunt=0;
// 15day analysis code 2--END




     total_cost_disp=total_cost;



//--- close the file 
   int file_size=0;
   file_size=FileSize(file_handle_2); 
   //Print("file size ", file_size);
   FileClose(file_handle_2);
   FileClose(file_handle_4);
   //if(file_size>10000 && CTi_Stru.hour>=12)
   //  {
   //   if(FileCopy("DayParameter.csv",0,"DayParameter2.csv",0)) 
   //      Print("File is copied!"); 
   //   else 
   //      Print("File is not copied!"); 
   //  }
   FileClose(file_handle_3);
   FileClose(file_handle); 
   First_tick=false;
   
   Posit_symb_cunt_disp=Posit_symb_cunt;
   Nega_symb_cunt_disp=Nega_symb_cunt;
   Zero_symb_cunt_disp=Zero_symb_cunt;
   Up_mos3_symb_cunt_disp=Up_mos3_symb_cunt;
   Low_manf3_symb_cunt_disp=Low_manf3_symb_cunt;
   
   Posit_symb_cunt=0;
   Nega_symb_cunt=0;
   Zero_symb_cunt=0;
   Up_mos3_symb_cunt=0;
   Low_manf3_symb_cunt=0;
   total_Current_BUY=0;
   total_Current_SELL=0;
   
   
//Print("End of first on timer");
  }
  
//+------------------------------------------------------------------+

//-----------------------------------------------END HosseinPRF------------------------------------------//

//--------------------------------------------------Functions-----------------------------------------------//
//--------------------------------------------Handel Create Function----------------------------------------//
bool CreateHandles(const string name_symbol,const ENUM_TIMEFRAMES timeframe,int &Hrsi
                   ,int &HMACD,int &HOBV,int &HSup_Res,int &Hforecastosc,int &HZigZag,int &HZigZag_2,int &HCandle,int &HStoch,int &HPTL)
  {
//--- create handle of the indicator iCustom
   Hrsi=iRSI(name_symbol,timeframe,14,applied_price);
   HMACD=iMACD(name_symbol,timeframe,fast_ema_period,slow_ema_period,signal_period,applied_price);
   HOBV=iOBV(name_symbol,timeframe,volume);
   HSup_Res=iSAR(name_symbol,timeframe,0.02,0.2);
   Hforecastosc=iCustom(name_symbol,timeframe,"forecastoscilator",length,t3,b,applied_price);
   HZigZag=iCustom(name_symbol,timeframe,"ZigZag",ExtDepth,ExtDeviation,ExtBackstep);////
   HZigZag_2=iCustom(name_symbol,timeframe,"ZigZag",ExtDepth2,ExtDeviation2,ExtBackstep2);////
   HCandle=iCustom(name_symbol,timeframe,"candlestick_type_color");////
   HStoch=iStochastic(name_symbol,timeframe,Kperiod,Dperiod,slowing,ma_method,price_field);
   HPTL= iCustom(name_symbol,timeframe,"PTL");


//--- if the handle is not created 
   if(Hrsi==INVALID_HANDLE || HMACD==INVALID_HANDLE || HOBV==INVALID_HANDLE || HSup_Res==INVALID_HANDLE
      || Hforecastosc==INVALID_HANDLE || HZigZag==INVALID_HANDLE || HZigZag_2==INVALID_HANDLE 
      || HCandle==INVALID_HANDLE || HStoch==INVALID_HANDLE|| HPTL==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code 
      PrintFormat("Failed to create handle of the one Of indicator in Function part of code %d",
                  name_symbol,
                  EnumToString(timeframe),
                  GetLastError());
      //--- the indicator is stopped early 
      return(false);
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+

void CalculateAllPositions(int &count_buys,int &count_sells)
  {
   count_buys=0;
   count_sells=0;
//   average_buy=0;
//   average_sell=0;
   for(int i=PositionsTotal()-1; i>=0; i--)
      if(m_Position.SelectByIndex(i)) // selects the position by index for further access to its properties
         //if(m_position.Symbol()==m_symbol.Name() && m_position.Magic()==InpMagic)
        {
         if(m_Position.PositionType()==POSITION_TYPE_BUY)
            count_buys++;
         if(m_Position.PositionType()==POSITION_TYPE_SELL)
            count_sells++;
        }
//---
   return;
  }
  
ENUM_TIMEFRAMES periodd (int t)
{
  if(t==0)        return(PERIOD_H1);
  else if(t==1)   return(PERIOD_H4);
  else if(t==2)   return(PERIOD_D1);   
  else                       return(PERIOD_CURRENT);
}  