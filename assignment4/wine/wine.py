# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:10:11 2025

@author: brooke
"""

import pandas as pd
import h5py
import numpy as np

# Load data

df = pd.read_csv(
    "wine.data",
    header=None
)

# Correct Wine Dataset column names (1 class + 13 attributes)
column_names = [
    "Class",
    "Alcohol",
    "Malic_Acid",
    "Ash",
    "Alcalinity_of_Ash",
    "Magnesium",
    "Total_Phenols",
    "Flavanoids",
    "Nonflavanoid_Phenols",
    "Proanthocyanins",
    "Color_Intensity",
    "Hue",
    "OD280_OD315",
    "Proline"
]

df.columns = column_names



# Create HDF5 structure and write dataset + metadata

hdf5_path = "wine_data_new.h5"

with h5py.File(hdf5_path, "w") as h5:

    # Create root-level group
    root_grp = h5.create_group("dataset")

    # Group-level metadata
    root_grp.attrs["source_file"] = "wine.data"
    root_grp.attrs["description"] = "Wine recognition chemical analysis dataset"
    root_grp.attrs["num_rows"] = df.shape[0]
    root_grp.attrs["num_columns"] = df.shape[1]

    root_grp.attrs["column_names"] = np.array(df.columns, dtype="S")
    dtype_strings = [str(df[c].dtype) for c in df.columns]
    root_grp.attrs["column_dtypes"] = np.array(dtype_strings, dtype="S")


    # Create dataset inside group with correct dtypes
    data_np = df.to_records(index=False)

    table = root_grp.create_dataset(
        "table",
        data=data_np,
        compression="gzip",
        compression_opts=9
    )

    # Add SAME METADATA to dataset so HDFView displays it
    table.attrs["source_file"] = "wine.data"
    table.attrs["description"] = "Wine recognition chemical analysis dataset"
    table.attrs["num_rows"] = df.shape[0]
    table.attrs["num_columns"] = df.shape[1]
    table.attrs["column_names"] = np.array(df.columns, dtype="S")
    table.attrs["column_dtypes"] = np.array(dtype_strings, dtype="S")
    

    # Read and parse the metadata from wine.names into sections
    with open("wine.names", "r") as f:
        raw = f.read()
    
    # split sections based on numbered headings
    def extract_section(text, start, end=None):
        start_idx = text.find(start)
        if start_idx == -1:
            return ""
        if end:
            end_idx = text.find(end)
            return text[start_idx:end_idx].strip() if end_idx != -1 else text[start_idx:].strip()
        return text[start_idx:].strip()
    
    # Extract sections by title
    title = extract_section(raw, "1. Title of Database:", "2. Sources:")
    sources = extract_section(raw, "2. Sources:", "3. Past Usage:")
    past_usage = extract_section(raw, "3. Past Usage:", "4. Relevant Information:")
    relevant_info = extract_section(raw, "4. Relevant Information:", "5. Number of Instances")
    num_instances = extract_section(raw, "5. Number of Instances", "6. Number of Attributes")
    num_attributes = extract_section(raw, "6. Number of Attributes", "7. For Each Attribute")
    attribute_notes = extract_section(raw, "7. For Each Attribute:", "8. Missing Attribute Values")
    missing_values = extract_section(raw, "8. Missing Attribute Values:", "9. Class Distribution")
    class_distribution = extract_section(raw, "9. Class Distribution:", None)
    
    # Store each parsed section as its own HDF5 attribute
    root_grp.attrs["title"] = title
    root_grp.attrs["sources"] = sources
    root_grp.attrs["past_usage"] = past_usage
    root_grp.attrs["relevant_information"] = relevant_info
    root_grp.attrs["num_instances"] = num_instances
    root_grp.attrs["num_attributes_info"] = num_attributes
    root_grp.attrs["attribute_notes"] = attribute_notes
    root_grp.attrs["missing_values_info"] = missing_values
    root_grp.attrs["class_distribution"] = class_distribution


print("HDF5 file created at:", hdf5_path)
