import os
import zipfile

'''Run this script only once'''

data_path = 'Data/HR_Lab_Data_Day_1'
zip_files = os.listdir(data_path)
unique_family = set()
 
for zip_file in zip_files:
    if not zip_file.lower().endswith('.zip'):
        continue
 
    zip_path = os.path.join(data_path, zip_file)
    family_id = zip_file[0:7]
    family_folder = os.path.join(data_path, family_id)
 
    if not os.path.exists(family_folder):
        os.mkdir(family_folder)
 
    # Get folder name for the condition
    filename_wo_ext = os.path.splitext(zip_file)[0]
    condition_folder = '_'.join(filename_wo_ext.split('_')[2:])
    extract_to = os.path.join(family_folder, condition_folder)

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
 
 
# Remove spaces in the folder name
for individual_sample in os.listdir(data_path):
    if individual_sample.endswith('.zip'):
        continue
 
    individual_path = os.path.join(data_path, individual_sample)
    if not os.path.isdir(individual_path):
        continue
 
    print (f"\nIndividual: {individual_sample}")
    
    for folder in os.listdir(individual_path):
        old_path = os.path.join(individual_path, folder)
        if os.path.isdir(old_path):
            new_name = folder.lower().replace(' ', '_').rstrip('_')
            new_path = os.path.join(individual_path, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {folder} -> {new_name}")