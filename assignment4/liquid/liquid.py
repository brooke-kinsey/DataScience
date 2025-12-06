# -*- coding: utf-8 -*-
"""
Full archival-grade HDF5 export
Includes: exact data replication, hierarchical structure,
and complete metadata import from Metadata.txt.
"""

import pandas as pd
import numpy as np
import h5py


# Loading CSV and fixing columns

df = pd.read_csv("liquid.csv")

clean_df = df.copy()

# Prepare column metadata for later storage
column_names = clean_df.columns.tolist()

# Convert all object dtypes to fixed-length byte strings
for col in clean_df.columns:
    if clean_df[col].dtype == object:
        # Convert to string first
        clean_df[col] = clean_df[col].astype(str)

        # Compute max byte length
        encoded = clean_df[col].str.encode("utf-8")
        max_len = encoded.str.len().max()
        if max_len is None or max_len < 1:
            max_len = 10

        # Store back as fixed-length bytes
        clean_df[col] = encoded.astype(f"S{max_len}")


column_dtypes = [str(dt) for dt in clean_df.dtypes]

# Create structured dtype
structured_dtype = np.dtype([
    (col, clean_df[col].dtype) for col in clean_df.columns
])

# Build records
records = np.zeros(len(clean_df), dtype=structured_dtype)
for col in clean_df.columns:
    records[col] = clean_df[col].values




# Reading metadata file

with open("Metadata.txt", "r", encoding="utf-8") as f:
    metadata_raw = f.read()

# Helper to extract sections
def extract_section(text, header):
    start = text.find(header)
    if start == -1:
        return ""
    start += len(header)
    remainder = text[start:]
    return remainder.strip()

# Parse by manual section boundaries based on the provided metadata file
general_info = metadata_raw.split("CONTEXTUAL INFORMATION")[0]
contextual_info = metadata_raw.split("CONTEXTUAL INFORMATION")[1].split("SHARING/ACCESS INFORMATION")[0]
sharing_access = metadata_raw.split("SHARING/ACCESS INFORMATION")[1].split("VERSIONING AND PROVENANCE")[0]
provenance = metadata_raw.split("VERSIONING AND PROVENANCE")[1].split("METHODOLOGICAL INFORMATION")[0]
methodology = metadata_raw.split("METHODOLOGICAL INFORMATION")[1].split("DATA & FILE OVERVIEW")[0]
file_overview = metadata_raw.split("DATA & FILE OVERVIEW")[1].split("TABULAR DATA-SPECIFIC INFORMATION")[0]
variable_info = metadata_raw.split("TABULAR DATA-SPECIFIC INFORMATION")[1]


# Creating HDF5 file and keeping hierarchy

hdf5_path = "liquid_data_new.h5"

with h5py.File(hdf5_path, "w") as h5:


    h5.attrs["source_file"] = "liquid.csv"
    h5.attrs["metadata_file"] = "Metadata.txt"
    h5.attrs["description"] = "Archival HDF5 version of liquid temperature dataset"
    h5.attrs["row_count"] = df.shape[0]
    h5.attrs["column_count"] = df.shape[1]
    h5.attrs["export_method"] = "Full archival conversion with metadata"


    dataset_grp = h5.create_group("dataset")
    data_grp = dataset_grp.create_group("data")
    metadata_grp = dataset_grp.create_group("metadata")


    # Write full dataset with dtypes

    table = data_grp.create_dataset(
        "table",
        data=records,
        dtype=structured_dtype,
        compression="gzip",
        compression_opts=9
    )

    # Dataset-level metadata
    table.attrs["column_names"] = np.array(column_names, dtype="S")
    table.attrs["column_dtypes"] = np.array(column_dtypes, dtype="S")
    table.attrs["num_rows"] = df.shape[0]
    table.attrs["num_columns"] = df.shape[1]


    # Storing structured metadata in groups

    # General Information 
    general = metadata_grp.create_group("general")
    general.attrs["raw_text"] = general_info

    # Contextual Information
    contextual = metadata_grp.create_group("contextual")
    contextual.attrs["raw_text"] = contextual_info

    # Sharing / Access Information
    sharing = metadata_grp.create_group("sharing")
    sharing.attrs["raw_text"] = sharing_access

    # Provenance / Versioning
    provenance_grp = metadata_grp.create_group("provenance")
    provenance_grp.attrs["raw_text"] = provenance

    # Methodological Information 
    methodology_grp = metadata_grp.create_group("methodology")
    methodology_grp.attrs["raw_text"] = methodology

    # File Overview 
    file_overview_grp = metadata_grp.create_group("file_overview")
    file_overview_grp.attrs["raw_text"] = file_overview

    # Variable Descriptions
    variable_grp = metadata_grp.create_group("variable_descriptions")
    variable_grp.attrs["raw_text"] = variable_info

    # Store entire cleaned metadata in one place as well
    metadata_grp.attrs["full_metadata_text"] = metadata_raw


    variable_grp.attrs["Starting_Temperature_Description"] = (
        "Starting temperature of the liquid. 37℉ for all liquids."
    )
    variable_grp.attrs["Final_Temperature_Description"] = (
        "Final temperature of the liquid. 70℉ for all liquids."
    )
    variable_grp.attrs["Time_Elapsed_Description"] = (
        "Time required for liquid to reach final temperature, in minutes/seconds."
    )
    variable_grp.attrs["Liquid_Description"] = (
        "Name/type of liquid measured."
    )

print("HDF5 successfully created:", hdf5_path)
