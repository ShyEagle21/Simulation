import pandas as pd
from datetime import datetime
import numpy as np
import math


def generate_demand(csv_file, sortation_center_id, predicted_volume, prediction_date):

    #predicted_volume = float(predicted_volume)
    data = pd.read_csv(csv_file)
    linehaul_schedule = pd.read_csv('linehaul_schedule.csv')
    mean_dict_actual_pack = {}
    std_dict_actual_pack = {}
    mean_dict_actual_pal = {}
    std_dict_actual_pal = {}
    mean_dict_per= {}
    std_dict_per = {}
    mean_dict_arrive = {}
    std_dict_arrive = {}

    filtered_df = data[data['sortation_center_id'] == sortation_center_id]

    for truck in filtered_df['Truck Number'].unique():
        truck_filtered_df = data[data['Truck Number'] == truck]
        #calulate the mean of actual packages
        mean_pack = truck_filtered_df['actual_packages'].mean()
        mean_pal = truck_filtered_df['actual_pallets'].mean()
        mean_per = truck_filtered_df['percentage contribution'].mean()
        #caluclate the standard deviation of actual packages
        std_pack = truck_filtered_df['actual_packages'].std()
        std_pal = truck_filtered_df['actual_pallets'].std()
        std_per = truck_filtered_df['percentage contribution'].std()
        mean_dict_actual_pack[truck] = mean_pack
        std_dict_actual_pack[truck] = std_pack
        mean_dict_actual_pal[truck] = mean_pal
        std_dict_actual_pal[truck] = std_pal
        mean_dict_per[truck] = mean_per
        std_dict_per[truck] = std_per


    #make a table of the mean and standard deviation of actual packages
    mean_dict_actual_pack = pd.DataFrame(mean_dict_actual_pack.items(), columns=['Truck Number', 'Mean Packages'])
    std_dict_actual_pack = pd.DataFrame(std_dict_actual_pack.items(), columns=['Truck Number', 'Standard Deviation Packages'])
    mean_dict_per = pd.DataFrame(mean_dict_per.items(), columns=['Truck Number', 'Mean Percentage'])
    std_dict_per = pd.DataFrame(std_dict_per.items(), columns=['Truck Number', 'Standard Deviation Percentage'])
    mean_dict_actual_pal = pd.DataFrame(mean_dict_actual_pal.items(), columns=['Truck Number', 'Mean Pallets'])
    std_dict_actual_pal = pd.DataFrame(std_dict_actual_pal.items(), columns=['Truck Number', 'Standard Deviation Pallets'])
    mean_std_actual = pd.merge(mean_dict_actual_pack, std_dict_actual_pack, on='Truck Number')
    mean_std_per = pd.merge(mean_dict_per, std_dict_per, on='Truck Number')
    mean_std_actual_pal = pd.merge(mean_dict_actual_pal, std_dict_actual_pal, on='Truck Number')
    mean_std_stuff = pd.merge(mean_std_actual, mean_std_actual_pal, on='Truck Number')
    mean_std_all = pd.merge(mean_std_stuff, mean_std_per, on='Truck Number')


    # Define the start time
    start_time = '16:30'


    # Convert 'planned_arrival_datetime' and 'actual_arrival_datetime' to datetime
    #filtered_df['planned_arrival_datetime'] = pd.to_datetime(filtered_df['planned_arrival_datetime'])
    filtered_df.loc[:, 'planned_arrival_datetime'] = pd.to_datetime(filtered_df['planned_arrival_datetime'])
    #filtered_df['actual_arrival_datetime'] = pd.to_datetime(filtered_df['actual_arrival_datetime'])
    filtered_df.loc[:, 'actual_arrival_datetime'] = pd.to_datetime(filtered_df['actual_arrival_datetime'])
    planned_truck = {}

    #pull out the planned arrival time for each truck number



    # Filter out rows where the actual arrival is more than 3 hours early or late
    time_diff = (filtered_df['actual_arrival_datetime'] - filtered_df['planned_arrival_datetime']).abs()
    filtered_df = filtered_df[time_diff <= pd.Timedelta(hours=3)]

    # Iterate over each unique truck number
    for truck in filtered_df['Truck Number'].unique():
        truck_df = filtered_df[filtered_df['Truck Number'] == truck].reset_index(drop=True)
        
        # Convert 'actual_arrival_datetime' to datetime
        truck_df['arrival_time'] = pd.to_datetime(truck_df['actual_arrival_datetime'])
        planned_truck[truck] = truck_df['planned_arrival_datetime']
        # Iterate over each row in the truck DataFrame
        for idx, row in truck_df.iterrows():
            arrival_time = row['arrival_time']
            if arrival_time.time() < datetime.strptime('15:00:00', '%H:%M:%S').time():
                arrival_time += pd.Timedelta(days=1)
            
            shift_time = pd.to_datetime(f"{row['inbound_date']} {start_time}")
            truck_df.at[idx, 'shift_time'] = shift_time
            truck_df.at[idx, 'arrival_time'] = arrival_time

        truck_df['arrival_mins'] = (truck_df['arrival_time'] - truck_df['shift_time']).dt.total_seconds() // 60

        # Calculate mean and standard deviation
        mean = truck_df['arrival_mins'].mean()
        std = truck_df['arrival_mins'].std()
        
        mean_dict_arrive[truck] = mean
        std_dict_arrive[truck] = std

        df_mean_dict_arrive = pd.DataFrame(mean_dict_arrive.items(), columns=['Truck Number', 'Mean'])
        df_std_dict_arrive = pd.DataFrame(std_dict_arrive.items(), columns=['Truck Number', 'STD'])
        mean_std_arrival = pd.merge(df_mean_dict_arrive, df_std_dict_arrive, on='Truck Number')



            
    #########################################################################################################

    df_truck_assumptions = mean_std_all
    df_truck_arrival = mean_std_arrival



    df_truck_assumptions['vol_actualization'] = np.random.normal(df_truck_assumptions['Mean Percentage'], df_truck_assumptions['Standard Deviation Percentage'])
    df_truck_assumptions.fillna(0)
    linehaul = df_truck_assumptions['vol_actualization'].sum()
    TFC = 1-linehaul
    df_truck_assumptions.loc[15,'vol_actualization'] = TFC

    df_package_distribution = pd.DataFrame(df_truck_assumptions[['Truck Number', 'vol_actualization']])

    df_truck_assumptions['Average Packages Per Pallet'] = df_truck_assumptions['Mean Packages'] / df_truck_assumptions['Mean Pallets']  

    df_pallet_assumptions = pd.DataFrame(df_truck_assumptions['Average Packages Per Pallet'])

    df_pallet_assumptions = df_pallet_assumptions.iloc[:-1]

    time = '16:30'

    df_package_distribution['predicted_truck_volume'] = df_package_distribution['vol_actualization'] * predicted_volume
    df_package_distribution = df_package_distribution.drop(15)
    df_package_distribution['predicted_truck_volume'] = df_package_distribution['predicted_truck_volume'].astype(int)
    df_package_distribution['predicted_truck_volume'] = df_package_distribution['predicted_truck_volume'].apply(lambda x: max(x, 0))
    df_package_distribution['Truck Number'] = df_package_distribution['Truck Number'].astype(int)

    total_packages = df_package_distribution['predicted_truck_volume'].sum()
    TFC_vol = predicted_volume - total_packages
    TFC_pallets = math.ceil(TFC_vol / 55)

    if predicted_volume != total_packages+TFC_vol:
        print(f'Error: Total packages ({total_packages}) do not match predicted volume({predicted_volume})')

    

    df_truck_assumptions['Scheduled Arrival Time'] = linehaul_schedule['Scheduled Arrival Time']    
    
    TFC_arrival = df_truck_assumptions.loc[15, 'Scheduled Arrival Time']

    

    start_time = pd.to_datetime(f'{prediction_date} {time}')

    TFC_arrival_dt = pd.to_datetime(prediction_date +" " + TFC_arrival)
    TFC_arrival_minutes = (TFC_arrival_dt - start_time).seconds // 60
        
    if TFC_arrival_minutes <0:
        TFC_arrival_minutes = 0

    df_package_distribution['arrival_actualization'] = np.random.normal(df_truck_arrival['Mean'], df_truck_arrival['STD'])

    for i in range(15):
        df_package_distribution.loc[i, 'pallets'] = math.ceil(df_package_distribution.loc[i, 'predicted_truck_volume'] / df_pallet_assumptions.loc[i,'Average Packages Per Pallet'])
    df_package_distribution['pallets'] = df_package_distribution['pallets'].astype(int)

    return df_package_distribution, TFC_vol, TFC_arrival_minutes, TFC_pallets