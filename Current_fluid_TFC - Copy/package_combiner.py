import os
import pandas as pd
def combine_and_convert_packages(folder_path, output_file):

    def correct_timestamps(row):
        timestamps = ['pkg_received_utc_ts', 'pkg_inducted_utc_ts', 'pkg_sorted_utc_ts', 'pkg_staged_utc_ts', 'pkg_loaded_utc_ts']
        for i in range(len(timestamps)-1):
            if pd.isnull(row[timestamps[i]]):
                row[timestamps[i]] = row[timestamps[i+1]]
            if row[timestamps[i]] > row[timestamps[i]]:
                row[timestamps[i]], row[timestamps[i+1]] = row[timestamps[i+1]], row[timestamps[i]]
        return row

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    dfs = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        time_columns = ['pkg_sort_code_assigned_utc_ts',
                        'pkg_label_created_utc_ts',
                        'pkg_received_utc_ts',
                        'pkg_inducted_utc_ts',
                        'pkg_sorted_utc_ts',
                        'pkg_staged_utc_ts',
                        'pkg_loaded_utc_ts',
                        'pkg_outbound_utc_ts',
                        'pkg_critical_pull_time_utc_ts']
        

        for col in time_columns:
            if col in df.columns:
                df[col]=pd.to_datetime(df[col], errors = 'coerce')

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index = True)

    #combined_df = combined_df[combined_df['pkg_cancelled_utc_ts'].isna()]

    combined_df.to_csv(output_file,index=False)
    print(f'comined CSV file saved as {output_file}')
    df_corrected = combined_df
    df_corrected = combined_df.apply(correct_timestamps, axis = 1)
    print("Timestamp Corrections Applied")
    #df_corrected = df_corrected.dropna()
    #df_corrected.set_index('scdre.package_tracking_number', inplace=True)
    

    return df_corrected