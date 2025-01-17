
####
####
####
####
import pandas as pd
from datetime import datetime
import numpy as np
import math
import random

def generate_demand(predicted_volume, fluid_ratio, prediction_date):

    linehaul_schedule = pd.read_csv('linehaul_schedule_demo_fluid.csv')

    # Number of fluid and pallet trucks
    num_fluid_trucks = linehaul_schedule['Fluid'].value_counts().get(1, 0)
    num_pallet_trucks = linehaul_schedule['Fluid'].value_counts().get(0, 0)
    total_trucks = num_fluid_trucks + num_pallet_trucks
    # Minimum volume for fluid trucks to ensure they carry more than 50% of the total volume
    mean = predicted_volume * fluid_ratio / num_fluid_trucks
    std_dev = .2 * mean

    
    # Distribute the volume for fluid trucks
    fluid_volumes = [random.gauss(mean / num_fluid_trucks, std_dev) for _ in range(num_fluid_trucks)]
    fluid_volumes = [math.ceil(volume) for volume in fluid_volumes]
    fluid_total = sum(fluid_volumes)

    if fluid_total < predicted_volume * fluid_ratio:
        needed_vol = predicted_volume*fluid_ratio-fluid_total
        needed_vol_per_truck = round(needed_vol/num_fluid_trucks)
        for i in range(num_fluid_trucks):
            fluid_volumes[i] += needed_vol_per_truck
        fluid_total = sum(fluid_volumes)
        if fluid_total < predicted_volume * fluid_ratio:
            fluid_volumes[0] += predicted_volume * fluid_ratio - fluid_total

        
    # Adjust the fluid volumes to ensure they sum up to more than 50% of the total volume
    
    # Remaining volume for pallet trucks
    remaining_volume = predicted_volume - fluid_total
    mean = remaining_volume / num_pallet_trucks
    std_dev = 0.2 * mean

    # Distribute the remaining volume among pallet trucks
    pallet_volumes = [random.gauss(mean, std_dev) for _ in range(num_pallet_trucks)]
    pallet_volumes = [math.ceil(volume) for volume in pallet_volumes]

    pallet_total = sum(pallet_volumes)

    # Adjust the volumes to match the remaining volume
    while pallet_total != remaining_volume:
        difference = remaining_volume - pallet_total
        adjustment_per_truck = difference // num_pallet_trucks
        remainder = difference % num_pallet_trucks

        for i in range(num_pallet_trucks):
            pallet_volumes[i] += adjustment_per_truck
            if i < remainder:
                pallet_volumes[i] += 1

        pallet_total = sum(pallet_volumes)

        # Adjust the pallet volumes to ensure they sum up to the remaining volume

       # Create truck numbers
    fluid_truck_numbers = linehaul_schedule[linehaul_schedule['Fluid'] == 1]['Truck Number'].tolist()
    pallet_truck_numbers = linehaul_schedule[linehaul_schedule['Fluid'] == 0]['Truck Number'].tolist()

    # Combine truck numbers with volumes
    fluid_data = {'Truck Number': fluid_truck_numbers, 'Volume': fluid_volumes}
    pallet_data = {'Truck Number': pallet_truck_numbers, 'Volume': pallet_volumes}

    # Create DataFrames
    fluid_df = pd.DataFrame(fluid_data)
    pallet_df = pd.DataFrame(pallet_data)

    # Combine both DataFrames
    combined_df = pd.concat([fluid_df, pallet_df], ignore_index=True)
    df_package_distribution = combined_df.sort_values(by='Truck Number').reset_index(drop=True)

    # Define the start time
    start_time = '16:30'

    mean_std_arrival = pd.read_csv('mean_std_arrival.csv')
            
    #########################################################################################################

    
    df_truck_arrival = mean_std_arrival

    time = '16:30'

    start_time = pd.to_datetime(f'{prediction_date} {time}')
        

    df_package_distribution['arrival_actualization'] = np.random.normal(df_truck_arrival['Mean'], df_truck_arrival['STD'])

    for i in range(total_trucks):
        df_package_distribution.loc[i, 'pallets'] = math.ceil(df_package_distribution.loc[i, 'Volume'] / 60)
    df_package_distribution['pallets'] = df_package_distribution['pallets'].astype(int)

    for truck in fluid_truck_numbers:
        df_package_distribution.loc[truck-1, 'pallets'] = 1

    return df_package_distribution
    