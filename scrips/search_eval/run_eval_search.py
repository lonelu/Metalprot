import os
import sys
import prody as pr
import numpy as np
#You can either add the python package path.
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot.search import search, search_eval
from metalprot.basic import filter
import pickle

'''
python /mnt/e/GitHub_Design/Metalprot/scrips/search_eval/run_eval_search.py
'''
### Please select the correct database based on the eval function you are using. 
query_dir = '/mnt/e/DesignData/ligands/ZN_rcsb_datesplit/20211013/20211013_selfcenter/pickle_all/'

with open(query_dir + 'all_metal_vdm.pkl', 'rb') as f:
    query_all_metal = pickle.load(f)

with open(query_dir + 'AAMetalPhiPsi.pkl', 'rb') as f:
    all_querys = pickle.load(f)

with open(query_dir + 'cluster_centroid_dict.pkl', 'rb') as f:
    cluster_centroid_dict = pickle.load(f)

print(len(all_querys))


### run Search_struct


# workdir = '/mnt/e/DesignData/ligands/LigandBB/MID1sc10/'

# outdir = workdir + 'output_eval/'

# target_path = workdir + '5od1_zn.pdb'

# win_filter = [34, 60, 64]

# workdir = '/mnt/e/DesignData/ligands/LigandBB/6dwv/'

# outdir = workdir + 'output_eval/'

# target_path = workdir + '6dwv_core.pdb'

# win_filter = [4, 6, 15]



workdir = '/mnt/e/DesignData/ligands/FE_rcsb/20211206/_Seq_core_3contact/'

outdir = workdir + 'output_eval/'

target_path = workdir + '1aui_FE_1.pdb'


win_filter = []

metal_metal_dist = 0.6

num_contact_vdms = [3]

allowed_aa_combinations = []  # or something like [['H', 'H', 'H']]

geometry_path = None


_filter = filter.Search_filter(filter_abple = False, filter_phipsi = True, max_phipsi_val = 25, 
    filter_vdm_score = False, min_vdm_score = 0, filter_vdm_count = False, min_vdm_clu_num = 20,
    after_search_filter_geometry = True, filter_based_geometry_structure = True, angle_tol = 15, aa_aa_tol = 0.5, aa_metal_tol = 0.5,
    pair_angle_range = [85, 130], pair_aa_aa_dist_range = [2.8, 4], pair_metal_aa_dist_range = None,
    after_search_filter_qt_clash = True, vdm_vdm_clash_dist = 2.0, vdm_bb_clash_dist = 1.7, 
    write_filtered_result = False, selfcenter_filter_member_phipsi=True)


ss = search_eval.Search_eval(target_path,  outdir, all_querys, cluster_centroid_dict, query_all_metal, 
    num_contact_vdms, metal_metal_dist, win_filter, validateOriginStruct = True, search_filter= _filter, geometry_path = None,
    density_radius = 0.6, allowed_aa_combinations = allowed_aa_combinations, eval_density = False, eval_mmdist = False)


ss.run_eval_selfcenter_search()
