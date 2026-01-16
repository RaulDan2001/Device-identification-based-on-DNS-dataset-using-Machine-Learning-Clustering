import pickle
import pandas as pd
import os

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
datasets_dir = os.path.join(project_dir, "datasets")
pkl_file = os.path.join(datasets_dir, "dns_dataset.pkl")
output_file = os.path.join(datasets_dir, "dns_dataset.xlsx")

try:
    #Load the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Convert to DataFrame (if it is not already)
    if isinstance(data, pd.DataFrame):
        print(f"DEBUG: pkl was {type(data)}")
        df = data
    elif isinstance(data, dict):
        print(f"DEBUG: pkl was {type(data)}")
        df = pd.DataFrame(data)
    elif isinstance(data, list):
        print(f"DEBUG: pkl was {type(data)}")
        df = pd.DataFrame(data)
    else:
        print(f"Unexpected data type: {type(data)}")
        exit(1)

    # # Export to Excel
    # df.to_excel(output_file, index=False, sheet_name="DNS Data")
    # print(f"Succesfully exported {len(df)} rows to {output_file}")
    # print(f"File saved in {os.path.abspath(output_file)}")

    # Export to CSV
    df.to_csv(output_file.replace('.xlsx', '.csv'), index=False)
    print(f"Successfully exported {len(df)} rows to {output_file.replace('.xlsx', '.csv')}")
    

except FileNotFoundError:
    print(f"Error: {pkl_file} not found in the current directory")
except Exception as e:
    print(f"Error: {str(e)}")