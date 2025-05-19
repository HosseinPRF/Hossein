//+------------------------------------------------------------------+
//| Sample MQL5 script to write input_buy.csv with sample data       |
//+------------------------------------------------------------------+
void OnStart()
{
   string filename = "input_buy.csv";
   
   // سر ستون‌ها (باید دقیقاً با مدل پایتون مطابقت داشته باشد)
   string header = "Ravand_TF 15 min=,Ravand_TF 1h=,Ravand_TF 4h=,Ravand_TF D=,RSI 15 min=,MACD_Status 15 min=,third_ravand_perc_TF 1h=,n_bar_Change_ravand_TF 15 min=,PTL_Status 15 min=,Ravand_Zstr_TF 15 min=,third_ravand_perc_TF 4h=,Supp_Z_TF 1h=,n_bar_Change_ravand_Zstr 15 min=,now_ravand_perc_TF 4h=,n_bar_Change_ravand_TF 4h=,RSI 4h=";
   
   // یک خط داده فرضی (دقت کنید کاما جداکننده ستون‌هاست)
   string data = "Ravand mobham_Shayad_Kanal,Ravand_Transient Soodi To Nozooli,Ravand mobham_Shayad_Kanal,Ravand_Transient Soodi To Nozooli,50,0.3,-1.5,2,1,2,3,1,5,1.5,4,40";
   
   int file_handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI);
   if(file_handle != INVALID_HANDLE)
   {
      FileWriteString(file_handle, header + "\n");
      FileWriteString(file_handle, data + "\n");
      FileClose(file_handle);
      Print("File ", filename, " created successfully.");
   }
   else
   {
      Print("Failed to open file ", filename);
   }
}
