####
####
####
####

import demand_generator_fluid_store_inbound as dg
import numpy as np
import pandas as pd
import random

def simulation_generator(predict):
    # Example feature values

    predicted_volume = predict
    #print(f'Predicted Total Volume: {predicted_volume}')

    df_package_distribution = dg.generate_demand(predicted_volume, 0.7, '2024-09-01')

    csv_file = 'carrier_breakdown.csv'
    distributions = pd.read_csv(csv_file)

    total_packages = predicted_volume
    df_pallet_formation = pd.DataFrame(df_package_distribution[['Truck Number','Volume', 'pallets']])

    # Determine the number of packages going to each organization based on the distribution
    carrier_packages = {}
    for index, row in distributions.iterrows():
        while True:
            value = int(np.random.normal(row["average_percent"], row["std_dev"]) * total_packages)
            if value >= 0:
                carrier_packages[row["carrier"]] = value
                break

    # Adjust the total to match the exact number of packages
    total_assigned_packages = sum(carrier_packages.values())
    if total_assigned_packages != total_packages:
        difference = total_packages - total_assigned_packages
        carrier_packages["TLMD"] += difference

    df_carrier_breakdown = pd.DataFrame(list(carrier_packages.items()), columns=['Organization', 'Packages'])


    def assign_packages_to_pallets(trucks_df, packages_df):
        result = []
        
        # Create a list of all packages
        all_packages = []
        for j in range(len(packages_df)):
            org = packages_df.loc[j, 'Organization']
            num_packages = packages_df.loc[j, 'Packages']
            all_packages.extend([org] * num_packages)
        
        # Shuffle the list of all packages
        np.random.shuffle(all_packages)
        start_index = 0
        for i in range(len(trucks_df)):
            truck_number = trucks_df.loc[i, 'Truck Number']
            num_pallets = trucks_df.loc[i, 'pallets']
            predicted_truck_volume = int(trucks_df.loc[i, 'Volume'])

            # Skip trucks with zero pallets
            if num_pallets <= 0:
                continue
            
            # Get the packages for the current truck
            truck_packages = all_packages[start_index:(start_index + predicted_truck_volume)]
            start_index += predicted_truck_volume
            # Create a list of pallets for the current truck
            truck_pallets = [[] for _ in range(num_pallets)]
            
            # Randomly assign packages to pallets on the current truck
            for package in truck_packages:
                pallet_index = np.random.randint(0, num_pallets)
                truck_pallets[pallet_index].append(package)
            
            # Count the number of packages per organization on each pallet
            pallet_counts = []
            for pallet in truck_pallets:
                counts = {org: pallet.count(org) for org in packages_df['Organization']}
                pallet_counts.append(counts)
            
            result.append({
                'Truck Number': truck_number,
                'pallets': pallet_counts
            })
        
        return result


    assigned_packages = assign_packages_to_pallets(df_pallet_formation, df_carrier_breakdown)


    # Initialize lists to store data for DataFrame
    truck_data = assigned_packages
    arrival_times_df = pd.DataFrame(df_package_distribution[['Truck Number', 'arrival_actualization']])
    # Initialize lists to store data for DataFrame
    pallet_numbers = []
    package_numbers = []
    arrival_times_list = []
    scac_list = []
    linehaul_list = []

    # Initialize package counter
    package_counter = 1

    # Initialize pallet counter
    pallet_counter = 1

    # Iterate over trucks and pallets to generate DataFrame data
    for truck in truck_data:
        truck_number = truck['Truck Number']
        arrival_time = float(arrival_times_df[arrival_times_df['Truck Number'] == truck_number]['arrival_actualization'].values)
        arrival_time = max(arrival_time, 0.0)
        # Determine linehaul value based on truck number
        if arrival_time < 190:
            linehaul = 'A'
        elif arrival_time < 800:
            linehaul = 'B'
        elif arrival_time < 1800:
            linehaul = 'C'
        else:
            linehaul = 'Unknown'  # Handle unexpected truck numbers
        
        for pallet in truck['pallets']:
            scac_values = []
            for org, num_packages in pallet.items():
                scac_values.extend([org] * num_packages)
            np.random.shuffle(scac_values)  # Shuffle SCAC values within the pallet
            for scac in scac_values:
                pallet_numbers.append(pallet_counter)
                package_numbers.append(f"PKG{package_counter:06d}")
                arrival_times_list.append(arrival_time)
                scac_list.append(scac)
                linehaul_list.append(linehaul)
                package_counter += 1
            pallet_counter += 1

    # Create DataFrame
    df = pd.DataFrame({
        'pkg_received_utc_ts': arrival_times_list,
        'package_tracking_number': package_numbers,
        'scac': scac_list,
        'Pallet': pallet_numbers,
        'Linehaul': linehaul_list
    })

    return df, df_package_distribution
    



    

