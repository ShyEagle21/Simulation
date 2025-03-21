import pandas as pd
import matplotlib.pyplot as plt
import os


################################################################################

def buildup_analysis(df_corrected):
    
###########################################################################################

    def adjust_custom_day(timestamp):
        if timestamp.time() >= pd.Timestamp('17:00:00').time():
            return timestamp.date()
        else:
            return (timestamp-pd.Timedelta(days=1)).date()

###########################################################################################

    filtered_df = df_corrected[df_corrected['sortation_center'] == 3858]
    #filtered_df = filtered_df[filtered_df['scdre.scac'].str.contains('TLMD', na = False)]
    filtered_df = filtered_df.drop(columns = ['sortation_center'])
    filtered_df = filtered_df.drop(columns = ['scdre.ship_d'])
    filtered_df = filtered_df.drop(columns = ['scdre.scac'])
    filtered_df = filtered_df.drop(columns = ['pkg_sort_code_assigned_utc_ts'])
    filtered_df = filtered_df.drop(columns = ['pkg_label_created_utc_ts'])
    filtered_df = filtered_df.drop(columns = ['scdre.location_id'])
    filtered_df = filtered_df.drop(columns = ['scdre.route_sort_c'])
    filtered_df = filtered_df.drop(columns = ['scdre.sortation_center_package_sort_c'])
    filtered_df = filtered_df.drop(columns = ['scdre.sortation_center_package_inbound_pallet_id'])
    filtered_df = filtered_df.drop(columns = ['scdre.sortation_center_package_sort_partition'])
    filtered_df = filtered_df.drop(columns = ['pkg_cancelled_utc_ts'])
    filtered_df = filtered_df.drop(columns = ['pkg_outbound_utc_ts'])
    filtered_df = filtered_df.drop(columns = ['pkg_critical_pull_time_utc_ts'])
    filtered_df = filtered_df.drop(columns = ['scdre.package_tracking_number'])



    df_long = filtered_df.melt(var_name='timestamp_type', value_name = 'timestamp', ignore_index = False)
    df_long.set_index('timestamp', inplace = True)

    df_long['count'] = 1
    pivot_table = df_long.pivot_table(index='timestamp', columns = 'timestamp_type', values = 'count', aggfunc = 'sum', fill_value = 0)

    pivot_table = pivot_table.reset_index()

    pivot_table['pkg_received_utc_ts_cum'] = pivot_table['pkg_received_utc_ts'].cumsum()


    pivot_table['Received'] = pivot_table['pkg_received_utc_ts_cum']


    pivot_table['timestamp']=pd.to_datetime(pivot_table['timestamp'], errors = 'coerce')
    pivot_table['custom_day'] = pivot_table['timestamp'].apply(adjust_custom_day)

    vertical_lines_1 = ['20:00:00', '20:30:00']
    vertical_lines_2 =['00:00:00', '00:15:00', '03:00:00', '06:30:00', '9:30:00', '10:00:00', '13:30:00', '13:45:00']

    pkg_received_utc_ts_cum = 0

    
    daily_dic = {}

    for day, group in pivot_table.groupby('custom_day'):
        plt.figure(figsize=(10,6))
        group['pkg_received_utc_ts_cum'] = group['pkg_received_utc_ts'].cumsum() 


        group['Received'] = group['pkg_received_utc_ts_cum']


        plt.plot(group['timestamp'], group['Received'], label = 'Received')
        #plt.axhline(y=max_induct, color='blue', linestyle='--', label = 'Max Induct')

        
        day_lines =  day + pd.Timedelta(days=1)

        for vlt in vertical_lines_1:
            vline = pd.Timestamp(f'{day} {vlt}')
            plt.axvline(x=vline, color = 'red', linestyle = '--')

        for vlt in vertical_lines_2:
            vline = pd.Timestamp(f'{day_lines} {vlt}')
            plt.axvline(x=vline, color = 'red', linestyle = '--')


        plt.xlabel('Time')
        plt.ylabel('Sort Inventory')
        plt.title(f'Inventory on {day_lines}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
        pkg_received_utc_ts_cum = group['Received'].iloc[-1]
        print(f'Total Received: {pkg_received_utc_ts_cum}')
        daily_dic[day] = pkg_received_utc_ts_cum

    return daily_dic

    