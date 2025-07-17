import pandas as pd
import numpy as np
import torch

def index_to_roi_id_one(index, df_atlas):
    """
    return corresponding ROI ID
    """
    return int(df_atlas.loc[index, 'ROI ID'])

def index_to_roi_id(index_tensor, df_atlas):
    """
    index_tensor: 1D torch tensor of indices
    df_atlas: pandas DataFrame with ROI ID mapping
    Returns: list of ROI IDs corresponding to each index
    """
    return [int(df_atlas.loc[i.item(), 'ROI ID']) for i in index_tensor]


def load_atlas():
    # Load the brain atlas labels with additional columns

    df = pd.read_csv('./data/schaefer_2018/Schaefer2018_100Parcels_17Networks_order_info.txt', sep='\t', header=None)
    data = np.array(df)
    labels = data[0::2] # extract "roi names, left or right hemisphere, and subnetwork names"
    extra_data = np.array([i[0].split(' ') for i in data[1::2]]) # extract "roi id, {x, y, z}, and color info"

    badROIs = [9, 14, 43, 60, 61, 78, 89, 93]  # 8 regions with bad data ("1-based" indexing)

    # labels = np.delete(labels, badROIs) # remove the bad ROIs
    # extra_data = np.delete(extra_data, badROIs, axis=0) # remove the bad ROIs
    # print(labels.shape)
    # print(extra_data.shape)

    ### Note that in schaefer atlas, the region names are not explicitly given:
    # e.g., ['LH', 'SomMotA', '2'], there is no explicit region name, only the hemisphere, subnetwork, and parcel index.
    split_labels=[]
    for label in labels:
        parts = label[0].split('_')[1:]
        if len(parts)==4:
            split_labels.append(parts)
        if len(parts) < 4: # some regions have no name info
            print('Region name is not explicitly given:', label)
            parts.insert(2, 'roi')
            split_labels.append(parts)

    df_1 = pd.DataFrame(split_labels, columns=['Hemisphere', 'Subnetwork', 'Region', 'Parcel index'])
    df_2 = pd.DataFrame(extra_data, columns=['ROI ID', 'R', 'G', 'B', 'Color'])

    # Merge df and df_atlas by extending columns
    df_atlas = pd.concat([df_1, df_2], axis=1)
    df_atlas.drop([i-1 for i in badROIs], inplace=True)   #drop bad ROIs from the 100 regions.
    df_atlas.reset_index(drop=True, inplace=True)
    # df_atlas.to_excel('Schaefer2018_92Parcels_17Networks_order_info.xlsx', index=False)
    return df_atlas

def load_roi_dict(df_atlas):
    df = df_atlas[['Hemisphere', 'Subnetwork', 'ROI ID']]

    roi_by_network_hemisphere = {}

    grouped = df.groupby(['Subnetwork', 'Hemisphere'])['ROI ID'].apply(list)

    for (subnet, hemi), roi_list in grouped.items():
        if subnet not in roi_by_network_hemisphere:
            roi_by_network_hemisphere[subnet] = {}
        roi_by_network_hemisphere[subnet][hemi] = roi_list

    return roi_by_network_hemisphere

def map_roi_id_subnetwork(roi_dict):
    roi_lookup = {}
    subnet_list = list(roi_dict.keys())  # subnet name to index mapping

    for subnet_idx, subnet in enumerate(subnet_list):
        hemi_dict = roi_dict[subnet]
        for hemi, roi_ids in hemi_dict.items():
            for roi in roi_ids:
                roi_int = int(roi)
                hemi_id = 0 if hemi == 'LH' else 1
                roi_lookup[roi_int] = (hemi_id, subnet_idx)
    print(roi_lookup)
    return roi_lookup

def built_node_meta(roi_lookup):
    # Step 3: Construct node_meta from roi_lookup
    badROIs = [9, 14, 43, 60, 61, 78, 89, 93]
    badROIs_set = set(badROIs)  

    N = max(roi_lookup.keys())  # assume ROI index starts at 1
    node_meta = torch.zeros(N, 2, dtype=torch.long)

    for roi_id in range(1, N + 1):
        if roi_id in badROIs_set:
            continue
        if roi_id in roi_lookup:
            node_meta[roi_id - 1] = torch.tensor(roi_lookup[roi_id])
        else:
            raise ValueError(f" Warning: ROI ID {roi_id} not in roi_lookup, filled with zeros")
    return node_meta

