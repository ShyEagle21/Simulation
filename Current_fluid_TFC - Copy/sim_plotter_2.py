import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

def sim_plotter(df, date_simulated_1):
    #df = pd.read_csv('final_packages.csv')
    df.drop(df.columns[0], axis=1, inplace=True)
    df_1 = df.drop(columns = ['current_queue','tracking_number','scac','partition','pallet_id'])
    df_2 = df[df['scac'].str.contains('TLMD', na = False)]
    df_2 = df.drop(columns = ['current_queue','tracking_number','scac','partition','pallet_id'])
    date_simulated = datetime.strptime(date_simulated_1, "%m/%d/%Y")

    def convert_to_datetime(minutes, shift_start, date_sim):
        shift_start_dt = datetime.strptime(shift_start, "%H:%M")
        shift_start_minutes = shift_start_dt.hour * 60 + shift_start_dt.minute
        total_minutes = shift_start_minutes + minutes
        
        if total_minutes >= 24 * 60:
            total_minutes -= 24 * 60
        
        if total_minutes < shift_start_minutes:
            date = date_sim + timedelta(days=1)
        else:
            date = date_sim
        
        result_datetime = datetime.combine(date.date(), datetime.min.time()) + timedelta(minutes=total_minutes)
        return result_datetime

    # Convert the timestamps
    shift_start = "16:30"
    df_1['received_ts'] = df_1['received_ts'].apply(lambda x: convert_to_datetime(x, shift_start,date_simulated))
    df_1['inducted_ts'] = df_1['inducted_ts'].apply(lambda x: convert_to_datetime(x, shift_start,date_simulated))
    df_1['sorted_ts'] = df_1['sorted_ts'].apply(lambda x: convert_to_datetime(x, shift_start,date_simulated))
    df_2['sorted_ts'] = df_2['sorted_ts'].apply(lambda x: convert_to_datetime(x, shift_start,date_simulated))

    df_long = df_1.melt(var_name='timestamp_type', value_name='timestamp', ignore_index=False)
    df_long.set_index('timestamp', inplace=True)

    df_long['count'] = 1
    pivot_table = df_long.pivot_table(index='timestamp', columns='timestamp_type', values='count', aggfunc='sum', fill_value=0)

    pivot_table = pivot_table.reset_index()

    pivot_table['timestamp'] = pd.to_datetime(pivot_table['timestamp'], errors='coerce')

    pivot_table['received_ts_cum'] = pivot_table['received_ts'].cumsum() - pivot_table['inducted_ts'].cumsum()
    pivot_table['Received'] = pivot_table['received_ts_cum']

    # Combine `timestamp` and `Received` into a new DataFrame
    combined_df = pd.DataFrame({
        'timestamp': pivot_table['timestamp'],
        'Received': pivot_table['Received']
    })

    # Set 'timestamp' as the index


    combined_df.set_index('timestamp', inplace=True)
    combined_df.to_csv(f'{date_simulated_1}_received_ts.csv')


    #induct
    df_long = df_1.melt(var_name='timestamp_type', value_name='timestamp', ignore_index=False)
    df_long.set_index('timestamp', inplace=True)

    df_long['count'] = 1
    pivot_table = df_long.pivot_table(index='timestamp', columns='timestamp_type', values='count', aggfunc='sum', fill_value=0)

    pivot_table = pivot_table.reset_index()

    pivot_table['timestamp'] = pd.to_datetime(pivot_table['timestamp'], errors='coerce')

    pivot_table['inducted_ts_cum'] = pivot_table['inducted_ts'].cumsum() - pivot_table['sorted_ts'].cumsum()
    pivot_table['Inducted'] = pivot_table['inducted_ts_cum']

    # Combine `timestamp` and `Received` into a new DataFrame
    combined_df_2 = pd.DataFrame({
        'timestamp': pivot_table['timestamp'],
        'Inducted': pivot_table['Inducted']
    })

    # Set 'timestamp' as the index


    combined_df_2.set_index('timestamp', inplace=True)
    combined_df_2.to_csv(f'{date_simulated_1}_inducted_ts.csv')

    #sort
    df_long = df_2.melt(var_name='timestamp_type', value_name='timestamp', ignore_index=False)
    df_long.set_index('timestamp', inplace=True)

    df_long['count'] = 1
    pivot_table = df_long.pivot_table(index='timestamp', columns='timestamp_type', values='count', aggfunc='sum', fill_value=0)

    pivot_table = pivot_table.reset_index()

    pivot_table['timestamp'] = pd.to_datetime(pivot_table['timestamp'], errors='coerce')


    pivot_table['sorted_ts_cum'] = pivot_table['sorted_ts'].cumsum()
    pivot_table['Sorted'] = pivot_table['sorted_ts_cum']

    # Combine `timestamp` and `Received` into a new DataFrame
    combined_df_3 = pd.DataFrame({
        'timestamp': pivot_table['timestamp'],
        'Sorted': pivot_table['Sorted']
    })

    # Set 'timestamp' as the index


    combined_df_3.set_index('timestamp', inplace=True)
    combined_df_3.to_csv(f'{date_simulated_1}_sorted_ts.csv')


    # Export to CSV
   
    #plt.plot(pivot_table['timestamp'], pivot_table['Received'], label='Received')
    # plt.axhline(y=max_induct, color='blue', linestyle='--', label='Max Induct')

    # for vlt in vertical_lines_1:
    #     vline = pd.Timestamp(f'{day} {vlt}')
    #     plt.axvline(x=vline, color='red', linestyle='--')

    # for vlt in vertical_lines_2:
    #     vline = pd.Timestamp(f'{day_lines} {vlt}')
    #     plt.axvline(x=vline, color='red', linestyle='--')

    #plt.xlabel('Time')
    #plt.ylabel('Sort Inventory')
    #plt.title('Simulated Loading Dock')
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
