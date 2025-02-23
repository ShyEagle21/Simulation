
import demand_generator as dg
import numpy as np
import pandas as pd
import random
import math

def simulation_generator(predict, partition_ratios):

    # Example feature values
    predicted_volume = predict
    df_package_distribution, TFC_vol, TFC_arrival_minutes, TFC_pallets = dg.generate_demand('linehaul_all_predict - Copy.csv', 3858, predicted_volume, '2024-09-01')
    csv_file = 'carrier_breakdown.csv'
    distributions = pd.read_csv(csv_file)
    total_packages = predicted_volume
    df_pallet_formation = pd.DataFrame(df_package_distribution[['Truck Number','predicted_truck_volume', 'pallets']])

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
        filtered_carriers = [key for key in carrier_packages.keys() if key != "FDE"]
        bonus_carrier = random.choice(filtered_carriers)
        carrier_packages[bonus_carrier] += difference
        carrier_packages['TLMD'] = carrier_packages['TLMD'] - TFC_vol

    df_carrier_breakdown = pd.DataFrame(list(carrier_packages.items()), columns=['Organization', 'Packages'])
    #take the value from the TFC and add it to the TLMD
    total_tlmd_volume = int(df_carrier_breakdown.loc[df_carrier_breakdown['Organization'] == 'TLMD', 'Packages'].iloc[0]) + TFC_vol
    #print(f'Total TLMD Volume: {total_tlmd_volume}')

    def assign_packages_to_pallets(trucks_df, packages_df):
        result = []
        tlmd_count = 0
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
            predicted_truck_volume = trucks_df.loc[i, 'predicted_truck_volume']
            
            # Skip trucks with zero pallets
            if num_pallets <= 0:
                continue
            
            # Get the packages for the current truck
            truck_packages = all_packages[start_index:start_index + predicted_truck_volume]
            start_index += predicted_truck_volume
            if i >= 11:
                count_tlmd = truck_packages.count('TLMD')
                if 12 <= truck_number <= 15:
                    tlmd_count += count_tlmd
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
        
        return result, tlmd_count
    
    def reallocate_packages_to_tlmd_pallets(truck_pallet_data):
        reallocated_result = []

        for truck_data in truck_pallet_data:
            truck_number = truck_data['Truck Number']
            pallets = truck_data['pallets']

            # Initialize new pallets in the desired format
            new_pallets = []

            if truck_number in [12, 13, 14, 15]:
                # Separate TLMD and non-TLMD packages
                tlmd_packages = []
                other_packages = {org: [] for org in pallets[0].keys() if org != 'TLMD'}

                for pallet in pallets:
                    for org, count in pallet.items():
                        if org == 'TLMD':
                            tlmd_packages.extend([org] * count)
                        else:
                            other_packages[org].extend([org] * count)

                # Calculate the number of TLMD pallets
                num_tlmd_pallets = max(1, 2 * len(tlmd_packages) // 45)  # At least 1 pallet for TLMD
                tlmd_packages_per_pallet = len(tlmd_packages) // num_tlmd_pallets
                remaining_tlmd_packages = len(tlmd_packages) % num_tlmd_pallets

                # Create TLMD pallets
                tlmd_pallets = []
                for i in range(num_tlmd_pallets):
                    count = tlmd_packages_per_pallet + (1 if i < remaining_tlmd_packages else 0)
                    tlmd_pallet = {org: 0 for org in pallets[0].keys()}
                    tlmd_pallet['TLMD'] = count
                    tlmd_pallets.append(tlmd_pallet)

                # Redistribute other packages proportionally across non-TLMD pallets
                total_other_packages = sum(len(other_packages[org]) for org in other_packages)
                num_other_pallets = max(1, 2 * total_other_packages // 45)  # At least 1 pallet for other packages
                other_pallets = [{org: 0 for org in pallets[0].keys()} for _ in range(num_other_pallets)]

                for org, org_packages in other_packages.items():
                    packages_per_pallet = len(org_packages) // num_other_pallets
                    remaining_packages = len(org_packages) % num_other_pallets

                    index = 0
                    for i in range(num_other_pallets):
                        count = packages_per_pallet + (1 if i < remaining_packages else 0)
                        other_pallets[i][org] = count
                        index += count

                # Combine TLMD and non-TLMD pallets
                new_pallets = tlmd_pallets + other_pallets
            else:
                # For other trucks, retain the original pallet structure
                new_pallets = pallets

            # Verify that the number of packages on each truck remains the same
            input_package_count = sum(sum(pallet.values()) for pallet in pallets)
            output_package_count = sum(sum(pallet.values()) for pallet in new_pallets)
            if input_package_count != output_package_count:
                raise ValueError(
                    f"Package count mismatch for Truck {truck_number}: "
                    f"Input = {input_package_count}, Output = {output_package_count}"
                )

            reallocated_result.append({
                'Truck Number': truck_number,
                'pallets': new_pallets
            })

        return reallocated_result




    assigned_packages_norm, TLMD_LHC_norm = assign_packages_to_pallets(df_pallet_formation, df_carrier_breakdown)
    assigned_packages_split = reallocate_packages_to_tlmd_pallets(assigned_packages_norm)
    new_entry_fluid = {'Truck Number': 16, 'pallets': [{'TLMD':TFC_vol, 'FDEG': 0, 'UPSN': 0, 'USPS': 0, 'FDE': 0}]}

    packages_per_pallet = 55
    pallets = []
    for i in range(TFC_pallets):
        if i < TFC_pallets - 1:
            # All pallets except the last one have 55 TLMD packages
            pallets.append({'TLMD': packages_per_pallet, 'FDEG': 0, 'UPSN': 0, 'USPS': 0, 'FDE': 0, 'NC': 0})
        else:
            # The last pallet takes the remaining volume
            remaining_packages = TFC_vol - (packages_per_pallet * (TFC_pallets - 1))
            pallets.append({'TLMD': remaining_packages, 'FDEG': 0, 'UPSN': 0, 'USPS': 0, 'FDE': 0, 'NC': 0})
    # Create the final entry
    new_entry_pallet = {
        'Truck Number': 16,
        'pallets': pallets
}
    new_entry_norm = new_entry_fluid.copy()
    new_entry_split = new_entry_fluid.copy()
    assigned_packages_norm.append(new_entry_norm) 
    assigned_packages_split.append(new_entry_split)
    #print(assigned_packages)




    LH_C_TLMD_norm = 0
    for truck in assigned_packages_norm:
        truck_number = truck['Truck Number']
        if 12 <= truck_number <= 15:
            for pallet in truck['pallets']:
                LH_C_TLMD_norm += pallet['TLMD']

    LH_C_TLMD_split = 0
    for truck in assigned_packages_split:
        truck_number = truck['Truck Number']
        if 12 <= truck_number <= 15:
            for pallet in truck['pallets']:
                LH_C_TLMD_split += pallet['TLMD']

    # Initialize lists to store data for DataFrame
    truck_data_norm = assigned_packages_norm
    truck_data_split = assigned_packages_split
    arrival_times_df = pd.DataFrame(df_package_distribution[['Truck Number', 'arrival_actualization']])
    # New row to add
    new_row = pd.DataFrame({'Truck Number': [16], 'arrival_actualization': [360]})

    # Adding the new row to the DataFrame using pd.concat
    arrival_times_df = pd.concat([arrival_times_df, new_row], ignore_index=True)
    # Initialize lists to store data for DataFrame
    # Initialize lists
    pallet_numbers = []
    package_numbers = []
    arrival_times_list = []
    scac_list = []
    linehaul_list = []
    partition_list = []

    # Initialize counters
    package_counter = 1
    pallet_counter = 1
    #print(f'calculation total_tlmd_volume: {total_tlmd_volume}')
    # Calculate partitions
    partition_1 = round(partition_ratios[0] * total_tlmd_volume)
    partition_2 = round(partition_ratios[1] * total_tlmd_volume)
    partition_3 = round(partition_ratios[2] * total_tlmd_volume)


    # Adjust partitions if they don't sum up correctly
    if partition_1 + partition_2 + partition_3 != total_tlmd_volume:
        partition_3 = total_tlmd_volume - partition_1 - partition_2

    
    partition_3AB = partition_3 - TLMD_LHC_norm
    Partition_3C = TLMD_LHC_norm

    #print(f'partition_1: {partition_1}, partition_2: {partition_2}, partition_3: {partition_3}')
    #print(f'Partition 3AB: {partition_3AB}, Partition 3C: {Partition_3C}')

    # Calculate partition limits

    

    # Create and shuffle partitions list
    partitions_non_split = (
        ['1'] * partition_1 +
        ['2'] * partition_2 +
        ['3AB'] * (partition_3 - TLMD_LHC_norm)
    )
    np.random.shuffle(partitions_non_split)
    
        # Create and shuffle partitions list
    partitions_split = (
        ['1'] * partition_1 +
        ['2'] * partition_2 +
        ['3AB'] * (partition_3 - TLMD_LHC_norm)
    )
    np.random.shuffle(partitions_non_split)
    np.random.shuffle(partitions_split)



    partition_counts = {'1': 0, '2': 0, '3AB': 0, '3C': 0}
    # Iterate over trucks and pallets to generate DataFrame data
    for truck in truck_data_norm:
        truck_number = truck['Truck Number']
        arrival_time = float(arrival_times_df[arrival_times_df['Truck Number'] == truck_number]['arrival_actualization'].values)
        arrival_time = max(arrival_time, 0.0)
        
        # Determine linehaul value based on truck number
        if 1 <= truck_number <= 6:
            linehaul = 'A'
        elif 7 <= truck_number <= 11:
            linehaul = 'B'
        elif 12 <= truck_number <= 15:
            linehaul = 'C'
        elif truck_number == 16:
            linehaul = 'TFC'
        else:
            linehaul = 'Unknown'

        for pallet in truck['pallets']:
            scac_values = []
            for org, num_packages in pallet.items():
                scac_values.extend([org] * num_packages)
            np.random.shuffle(scac_values)
            
            for scac in scac_values:
                pallet_numbers.append(pallet_counter)
                package_numbers.append(f"PKG{package_counter:06d}")
                arrival_times_list.append(arrival_time)
                scac_list.append(scac)
                linehaul_list.append(linehaul)
                
                # Assign partition based on SCAC and linehaul
                if scac in ['USPS', 'UPSN', 'FDEG', 'FDE']:
                    partition_list.append(scac)
                elif scac == 'TLMD':
                    if linehaul in ['A', 'B', 'TFC']:
                        partition = partitions_non_split.pop(0)
                        partition_list.append(partition)
                        partition_counts[partition] += 1
                    elif linehaul == 'C':
                        partition_list.append('3C')
                        partition_counts['3C'] += 1
                    else:
                        partition_list.append('Unknown')
                package_counter += 1
            pallet_counter += 1
    # Create DataFrame
    df_norm = pd.DataFrame({
        'pkg_received_utc_ts': arrival_times_list,
        'package_tracking_number': package_numbers,
        'scac': scac_list,
        'Pallet': pallet_numbers,
        'Linehaul': linehaul_list,
        'Partition': partition_list
    })
    #print('Finished Norm')
# Iterate over trucks and pallets to generate DataFrame data
    pallet_numbers = []
    package_numbers = []
    arrival_times_list = []
    scac_list = []
    linehaul_list = []
    partition_list = []

    # Initialize counters
    package_counter = 1
    pallet_counter = 1
    partition_counts = {'1': 0, '2': 0, '3AB': 0, '3C': 0}
    for truck in truck_data_split:
        truck_number = truck['Truck Number']
        arrival_time = float(arrival_times_df[arrival_times_df['Truck Number'] == truck_number]['arrival_actualization'].values)
        arrival_time = max(arrival_time, 0.0)
        
        # Determine linehaul value based on truck number
        if 1 <= truck_number <= 6:
            linehaul = 'A'
        elif 7 <= truck_number <= 11:
            linehaul = 'B'
        elif 12 <= truck_number <= 15:
            linehaul = 'C'
        elif truck_number == 16:
            linehaul = 'TFC'
        else:
            linehaul = 'Unknown'

        for pallet in truck['pallets']:
            scac_values = []
            for org, num_packages in pallet.items():
                scac_values.extend([org] * num_packages)
            np.random.shuffle(scac_values)
            
            for scac in scac_values:
                pallet_numbers.append(pallet_counter)
                package_numbers.append(f"PKG{package_counter:06d}")
                arrival_times_list.append(arrival_time)
                scac_list.append(scac)
                linehaul_list.append(linehaul)
                
                # Assign partition based on SCAC and linehaul
                if scac in ['USPS', 'UPSN', 'FDEG', 'FDE']:
                    partition_list.append(scac)
                elif scac == 'TLMD':
                    if linehaul in ['A', 'B', 'TFC']:
                        partition = partitions_split.pop(0)
                        partition_list.append(partition)
                        partition_counts[partition] += 1
                    elif linehaul == 'C':
                        partition_list.append('3C')
                        partition_counts['3C'] += 1
                    else:
                        partition_list.append('Unknown')
                package_counter += 1
            pallet_counter += 1

    # Create DataFrame
    df_split = pd.DataFrame({
        'pkg_received_utc_ts': arrival_times_list,
        'package_tracking_number': package_numbers,
        'scac': scac_list,
        'Pallet': pallet_numbers,
        'Linehaul': linehaul_list,
        'Partition': partition_list
    })
    #print('Finished Split')
    return df_norm, df_split, df_package_distribution, TFC_arrival_minutes, partition_1, partition_2, partition_3AB, Partition_3C, partition_counts