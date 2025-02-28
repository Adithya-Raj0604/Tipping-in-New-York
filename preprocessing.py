import pandas as pd
import glob
import pyarrow.parquet as pq
import os

# Define the base directory path dynamically (relative path)
base_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_folder, "NYCdata")

# Get a list of all Parquet files, excluding 'combined_sample_data.parquet'
parquet_files = [f for f in glob.glob(os.path.join(data_folder, "*.parquet")) if "combined_sample_data.parquet" not in f]

print(f"Total files found: {len(parquet_files)}")
if not parquet_files:
    raise FileNotFoundError("No Parquet files found. Please check the directory path.")

# Load and sample Parquet files separately, then combine the samples
dfs = []
for file in parquet_files:
    try:
        print(f"\nProcessing file: {file}")
        df = pd.read_parquet(file, engine='pyarrow')  # Use PyArrow for efficiency

        # Check original row count
        original_rows = len(df)
        
        # Skip rows where 'hvfhs_license_num' is 'HV0005'
        df = df[df['hvfhs_license_num'] != 'HV0005']

        # Define essential columns (all except 'airport_fee' and 'wav_match_flag')
        essential_columns = [col for col in df.columns if col not in ['airport_fee', 'wav_match_flag']]
        
        # Drop rows where essential columns are missing
        df = df.dropna(subset=essential_columns)

        # Remove duplicate rows
        df = df.drop_duplicates()

        cleaned_rows = len(df)
        print(f"Original rows: {original_rows}, After dropping NA, duplicates, and HV0005: {cleaned_rows}")

        # If dataset is too small, do not include it
        if cleaned_rows < 100:
            print(f"Skipping file {file} - Too few valid rows after cleaning.")
            continue

        # Take 10% sample or at least 1000 rows, whichever is smaller
        sample_size = max(int(0.1 * cleaned_rows), min(cleaned_rows, 1000))
        sample_df = df.sample(n=sample_size, random_state=42)

        print(f"Sampled {len(sample_df)} rows from {file}")
        dfs.append(sample_df)

    except MemoryError as e:
        print(f"Memory error while loading {file}: {e}")

# Ensure dataframes were loaded
if not dfs:
    raise RuntimeError("No files could be loaded due to memory issues.")

# Concatenate sampled dataframes
combined_sample_df = pd.concat(dfs, ignore_index=True)

print(f"\nFinal combined sample dataset has {combined_sample_df.shape[0]} rows and {combined_sample_df.shape[1]} columns.")

# Define output file path
output_file = os.path.join(data_folder, "combined_sample_data.parquet")

# Remove existing file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print("Existing 'combined_sample_data.parquet' file removed.")

# Save the combined sampled file using PyArrow
combined_sample_df.to_parquet(output_file, index=False, engine='pyarrow')
print("Combined sample dataset saved as 'combined_sample_data.parquet'.")

# Process the CSV file
csv_file_path = os.path.join(base_folder, "NYC_Weather_2016_2022.csv")
cleaned_csv_file_path = os.path.join(base_folder, "NYC_Weather_2016_2022_cleaned.csv")
sampled_csv_file_path = os.path.join(base_folder, "NYC_Weather_2016_2022_sampled.csv")

# Load CSV file
print(f"\nProcessing CSV file: {csv_file_path}")
df_csv = pd.read_csv(csv_file_path)

# Drop rows where any column is missing (check all columns for NA values)
df_csv.dropna(inplace=True)

# Remove duplicate rows
df_csv.drop_duplicates(inplace=True)

cleaned_rows_csv = len(df_csv)
print(f"Original CSV rows: {len(df_csv)}, After dropping all rows with any NA and duplicates: {cleaned_rows_csv}")

# Save cleaned CSV file (overwrite if exists)
if os.path.exists(cleaned_csv_file_path):
    os.remove(cleaned_csv_file_path)
    print("Existing cleaned CSV file removed.")
df_csv.to_csv(cleaned_csv_file_path, index=False)
print("Cleaned CSV file saved.")

# Create a 10% sample of the CSV file
sample_size_csv = max(int(0.1 * cleaned_rows_csv), min(cleaned_rows_csv, 1000))
sampled_df_csv = df_csv.sample(n=sample_size_csv, random_state=42)

# Save sampled CSV file (overwrite if exists)
if os.path.exists(sampled_csv_file_path):
    os.remove(sampled_csv_file_path)
    print("Existing sampled CSV file removed.")
sampled_df_csv.to_csv(sampled_csv_file_path, index=False)
print("Sampled CSV file saved.")
