//+------------------------------------------------------------------+
//| ZigZagTrendAnalyzer.mqh                                          |
//| Class to analyze ZigZag trend, supports and resistances          |
//+------------------------------------------------------------------+
class ZigZagTrendAnalyzer
  {
private:
   double &ZigZag_Buffer_Zstr[];
   double &ZigZag_Buffer_F_Zstr[][2];
   double &ZigZag_Buffer_F2_Zstr[][2];
   int copied;

   bool &Strong_Signal_From_zigzag_Zstr[];
   bool &Signal_From_zigzag_Zstr[];
   int &n_bar_Change_ravand_Zstr[];
   string &Ravand_symbol_Zstr[];

   double last_price;
   double Ave__UptoDo_mosb_Zst;
   double chang_ravand_INDX_Zst;

public:
   ZigZagTrendAnalyzer(double &zigzag_buf[], double &zigzag_buf_f[][2], double &zigzag_buf_f2[][2],
                       int total_copied,
                       bool &strong_signal_arr[], bool &signal_arr[], int &n_bar_change_arr[], string &ravand_symbol_arr[])
     : ZigZag_Buffer_Zstr(zigzag_buf),
       ZigZag_Buffer_F_Zstr(zigzag_buf_f),
       ZigZag_Buffer_F2_Zstr(zigzag_buf_f2),
       copied(total_copied),
       Strong_Signal_From_zigzag_Zstr(strong_signal_arr),
       Signal_From_zigzag_Zstr(signal_arr),
       n_bar_Change_ravand_Zstr(n_bar_change_arr),
       Ravand_symbol_Zstr(ravand_symbol_arr)
     {
      last_price=0;
      Ave__UptoDo_mosb_Zst=0;
      chang_ravand_INDX_Zst=0;
     }

   void SetLastPrice(double price)
     {
      last_price=price;
     }

   void Process(int i, int symbol_size, double this_day_CP[], MqlRates mrate[])
     {
      if(last_price==0 && i<symbol_size)
         last_price=this_day_CP[i];
      if(i>=symbol_size && last_price==0)
         last_price=mrate[0].close;

      Strong_Signal_From_zigzag_Zstr[i]=false;
      if(ZigZag_Buffer_Zstr[0]==0 && ZigZag_Buffer_Zstr[1]!=0)
         Strong_Signal_From_zigzag_Zstr[i]=true;

      n_bar_Change_ravand_Zstr[i]=0;
      int n=0;

      if(ZigZag_Buffer_Zstr[0]==0)
        {
         Signal_From_zigzag_Zstr[i]=false;
         for(int k=0;k<copied;k++)
           {
            if(ZigZag_Buffer_Zstr[k]!=0)
              {
               ArrayResize(ZigZag_Buffer_F_Zstr,n+1,10);
               ZigZag_Buffer_F_Zstr[n][0]=ZigZag_Buffer_Zstr[k];
               ZigZag_Buffer_F_Zstr[n][1]=k;
               if(n==0)
                 n_bar_Change_ravand_Zstr[i]=k;
               n++;
              }
           }
        }
      else
        {
         Signal_From_zigzag_Zstr[i]=true;
         for(int k=1;k<copied;k++)
           {
            if(ZigZag_Buffer_Zstr[k]!=0)
              {
               ArrayResize(ZigZag_Buffer_F_Zstr,n+1,10);
               ZigZag_Buffer_F_Zstr[n][0]=ZigZag_Buffer_Zstr[k];
               ZigZag_Buffer_F_Zstr[n][1]=k;
               if(n==0)
                 n_bar_Change_ravand_Zstr[i]=k;
               n++;
              }
           }
        }

      int count_Zig_buff_Zst=ArrayRange(ZigZag_Buffer_F_Zstr,0);
      double shib_ravand_Zstr[];
      ArrayResize(shib_ravand_Zstr,count_Zig_buff_Zst,10);

      if(count_Zig_buff_Zst>=4)
        {
         double Sum_shib_rav_mosbat_Zst_Zst=0;
         double Sum_shib_rav_manfi_Zst=0;
         double Ave_shib_rav_mosbat_Zst=0;
         double Ave_shib_rav_manfi_Zst=0;
         double sum_UptoDo_mosb_Zst=0;
         double Ave__UptoDo_mosb_Zst_local=0;
         double sum_UptoDo_manf_Zst=0;
         double Ave__UptoDo_manf_Zst=0;
         int c=0;
         int d=0;

         for(int m=1;m<(count_Zig_buff_Zst);m++)
           {
            shib_ravand_Zstr[m]=
               (((ZigZag_Buffer_F_Zstr[m-1][0]-ZigZag_Buffer_F_Zstr[m][0])/ZigZag_Buffer_F_Zstr[m-1][0])*100)/
               (ZigZag_Buffer_F_Zstr[m][1]-ZigZag_Buffer_F_Zstr[m-1][1]);
           }
         shib_ravand_Zstr[0]=
            (((last_price-ZigZag_Buffer_F_Zstr[0][0])/last_price)*100)/ZigZag_Buffer_F_Zstr[0][1];

         int loop_limit=(count_Zig_buff_Zst>20)?19:(count_Zig_buff_Zst-1);

         for(int n=1;n<loop_limit;n++)
           {
            if(shib_ravand_Zstr[n]>0)
              {
               c++;
               Sum_shib_rav_mosbat_Zst_Zst+=shib_ravand_Zstr[n];
               Ave_shib_rav_mosbat_Zst=Sum_shib_rav_mosbat_Zst_Zst/c;
               sum_UptoDo_mosb_Zst+=
                  (((ZigZag_Buffer_F_Zstr[n-1][0]-ZigZag_Buffer_F_Zstr[n][0])/ZigZag_Buffer_F_Zstr[n-1][0])*100);
               Ave__UptoDo_mosb_Zst_local=sum_UptoDo_mosb_Zst/c;
              }
            else if(shib_ravand_Zstr[n]<0)
              {
               d++;
               Sum_shib_rav_manfi_Zst+=shib_ravand_Zstr[n];
               Ave_shib_rav_manfi_Zst=Sum_shib_rav_manfi_Zst/d;
               sum_UptoDo_manf_Zst+=
                  (((ZigZag_Buffer_F_Zstr[n-1][0]-ZigZag_Buffer_F_Zstr[n][0])/ZigZag_Buffer_F_Zstr[n-1][0])*100);
               Ave__UptoDo_manf_Zst=sum_UptoDo_manf_Zst/d;
              }
           }
         Ave__UptoDo_mosb_Zst=Ave__UptoDo_mosb_Zst_local;
        }
      else
        {
         return;
        }

      chang_ravand_INDX_Zst = Ave__UptoDo_mosb_Zst / 6;

      string Ravand_Zstr="";
      if(Strong_Signal_From_zigzag_Zstr[i]==false)
        {
         if(shib_ravand_Zstr[0]>0)
           Ravand_Zstr="Ravand Soodi";
         else if(shib_ravand_Zstr[0]<0)
           Ravand_Zstr="Ravand Nozooli";
         else
           Ravand_Zstr="Ravand mobham_Shayad_Kanal";
        }
      else
        {
         if(last_price>ZigZag_Buffer_F_Zstr[1][0] && (((last_price-ZigZag_Buffer_F_Zstr[1][0])/last_price)*100)>=chang_ravand_INDX_Zst)
           Ravand_Zstr="Ravand_Transient Soodi To Nozooli";
         else if(last_price<ZigZag_Buffer_F_Zstr[1][0] && (((last_price-ZigZag_Buffer_F_Zstr[1][0])/last_price)*100)<=-1*chang_ravand_INDX_Zst)
           Ravand_Zstr="Ravand_Transient Nozooli To Soodi";
         else
           Ravand_Zstr="Ravand_Transient_Ravand_Mobham";
        }

      if(ZigZag_Buffer_Zstr[0]==0 && ZigZag_Buffer_Zstr[1]==0)
         Ravand_Zstr="Ravand mobham_Shayad_Kanal";

      Ravand_symbol_Zstr[i]=Ravand_Zstr;
     }

   struct SupportResistancePoint
     {
      double price;
      double distance_percent;
     };

   SupportResistancePoint above_points[];
   SupportResistancePoint below_points[];
   int count_above=0;
   int count_below=0;

   void CalculateSupportResistance()
     {
      int points_count=ArrayRange(ZigZag_Buffer_F2_Zstr,0);

      ArrayResize(above_points,points_count);
      ArrayResize(below_points,points_count);

      count_above=0;
      count_below=0;

      for(int idx=0; idx<points_count; idx++)
        {
         double p=ZigZag_Buffer_F2_Zstr[idx][0];
         double dist=MathAbs(p-last_price)/last_price*100.0;

         if(p>last_price)
           {
            above_points[count_above].price=p;
            above_points[count_above].distance_percent=dist;
            count_above++;
           }
         else if(p<last_price)
           {
            below_points[count_below].price=p;
            below_points[count_below].distance_percent=dist;
            count_below++;
           }
        }

      ArrayResize(above_points,count_above);
      ArrayResize(below_points,count_below);

      SortPointsByDistance(above_points,count_above);
      SortPointsByDistance(below_points,count_below);
     }

   void SortPointsByDistance(SupportResistancePoint &arr[], int size)
     {
      for(int i=0; i<size-1; i++)
        {
         for(int j=i+1; j<size; j++)
           {
            if(arr[i].distance_percent > arr[j].distance_percent)
              {
               SupportResistancePoint temp=arr[i];
               arr[i]=arr[j];
               arr[j]=temp;
              }
           }
        }
     }

   int GetSupportCount() { return count_below; }
   int GetResistanceCount() { return count_above; }

   double GetSupportPrice(int idx) { if(idx>=0 && idx<count_below) return below_points[idx].price; return 0; }
   double GetSupportDistance(int idx) { if(idx>=0 && idx<count_below) return below_points[idx].distance_percent; return 0; }

   double GetResistancePrice(int idx) { if(idx>=0 && idx<count_above) return above_points[idx].price; return 0; }
   double GetResistanceDistance(int idx) { if(idx>=0 && idx<count_above) return above_points[idx].distance_percent; return 0; }
  };
