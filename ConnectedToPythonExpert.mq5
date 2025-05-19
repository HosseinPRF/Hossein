#include <WinUser32.mqh>

void OnStart()
{
    string inputFile = "input_buy.csv";
    string outputFile = "prediction_result.txt";
    string pythonExe = "C:\\Python39\\python.exe";       // مسیر python.exe را به سیستم خودت تغییر بده
    string pythonScript = "C:\\path\\to\\predict_script.py"; // مسیر اسکریپت پایتون را تغییر بده

    // 1. ساخت فایل ورودی input_buy.csv
    string csvData = 
        "Ravand_TF 15 min=,Ravand_TF 1h=,Ravand_TF 4h=,Ravand_TF D=,Ravand_Zstr_TF 15 min=,Ravand_Zstr_TF 1h=,Ravand_Zstr_TF 4h=,Ravand_Zstr_TF D=,PTL_signal 15 min=,PTL_signal 1h=,PTL_signal 4h=,PTL_signal D=,PTL_Status 15 min=,PTL_Status 1h=,PTL_Status 4h=,PTL_Status D=,RSI_Status 15 min=,RSI_Status 1h=,RSI_Status 4h=,RSI_Status D=,RSI 15 min=,RSI 1h=,RSI 4h=,RSI D=,MACD_Status 15 min=,MACD_Status 1h=,MACD_Status 4h=,MACD_Status D=,Stochastic_Status 15 min=,Stochastic_Status 1h=,Stochastic_Status 4h=,Stochastic_Status D=,Supp_Z_TF 15 min=,Supp_Z_TF 1h=,Supp_Z_TF 4h=,Supp_Z_TF D=,Ress_Z_TF 15 min=,Ress_Z_TF 1h=,Ress_Z_TF 4h=,Ress_Z_TF D=,n_bar_Change_ravand_TF 15 min=,n_bar_Change_ravand_TF 1h=,n_bar_Change_ravand_TF 4h=,n_bar_Change_ravand_TF D=,n_bar_Change_ravand_Zstr 15 min=,n_bar_Change_ravand_Zstr 1h=,n_bar_Change_ravand_Zstr 4h=,n_bar_Change_ravand_Zstr D=,now_ravand_perc_TF 15 min=,now_ravand_perc_TF 1h=,now_ravand_perc_TF 4h=,now_ravand_perc_TF D=,sec_ravand_perc_TF 15 min=,sec_ravand_perc_TF 1h=,sec_ravand_perc_TF 4h=,sec_ravand_perc_TF D=,third_ravand_perc_TF 15 min=,third_ravand_perc_TF 1h=,third_ravand_perc_TF 4h=,third_ravand_perc_TF D=\n"
        "Ravand mobham_Shayad_Kanal,Ravand_Transient Soodi To Nozooli,Ravand mobham_Shayad_Kanal,Ravand_Transient Soodi To Nozooli,Ravand mobham_Shayad_Kanal,Ravand_Transient Soodi To Nozooli,Ravand mobham_Shayad_Kanal,Ravand_Transient Soodi To Nozooli,No_newSignal,Buy Signa_StartSouod,No_newSignal,No_newSignal,PTL_isSouodi,PTL_isNeutral,PTL_isSouodi,PTL_isSouodi,Neutral,Neutral,Neutral,OB_Ehtiait_Nozool,51.35,55.77,65.36,74.30,Edame_Ravand_Souod,Edame_Ravand_Nozool,Ehtemale_Start_Nozool,Edame_Ravand_Souod,Neutral,Neutral,Neutral,Neutral,0,1.12,0,1.11,0,1.13,0,1.14,2,1,4,1,2,15,4,1,0,0,0,0,0.69,5.13,5.45,6.90,-1.18,-1.63,-2.35,-2.02";

    int file_handle = FileOpen(inputFile, FILE_WRITE|FILE_CSV|FILE_ANSI);
    if(file_handle != INVALID_HANDLE)
    {
        FileWriteString(file_handle, csvData);
        FileClose(file_handle);
        Print("فایل ورودی ساخته شد: ", inputFile);
    }
    else
    {
        Print("خطا در ساخت فایل ورودی");
        return;
    }

    // 2. اجرای اسکریپت پایتون
    string params = pythonScript;
    long ret = ShellExecuteW(0, "open", pythonExe, params, NULL, SW_HIDE);
    if(ret <= 32)
    {
        Print("خطا در اجرای اسکریپت پایتون");
        return;
    }
    Print("اسکریپت پایتون اجرا شد");

    // 3. انتظار برای تولید خروجی
    Sleep(2000);  // 2 ثانیه صبر کن (می‌تونی بهبود بدی با چک کردن وجود فایل)

    // 4. خواندن خروجی
    int file_out = FileOpen(outputFile, FILE_READ|FILE_ANSI);
    if(file_out == INVALID_HANDLE)
    {
        Print("خطا در خواندن فایل خروجی");
        return;
    }

    string result = FileReadString(file_out);
    FileClose(file_out);
    Print("احتمال سوددهی از پایتون: ", result);

    double prob = StringToDouble(result);
    if(prob > 0.5)
        Print("تصمیم: خرید");
    else
        Print("تصمیم: خرید نکن");

    // در اینجا می‌تونی بر اساس prob پوزیشن باز/بسته کنی
}
