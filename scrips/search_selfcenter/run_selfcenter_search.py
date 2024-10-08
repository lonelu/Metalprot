#You can either add the python package path.
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot.search import search_selfcenter
from metalprot.basic import filter
import pickle
import time
import prody as pr

'''
python /Users/lonelu/GitHub_Design/Metalprot/scrips/search_selfcenter/run_selfcenter_search.py

'''
start_time = time.time()

query_dir = '/Users/lonelu/GitHub_Design/Metalprot/Database/ZN_HDEC_20210113/'

with open(query_dir + 'all_metal_vdm.pkl', 'rb') as f:
    query_all_metal = pickle.load(f)

with open(query_dir + 'AAMetalPhiPsi.pkl', 'rb') as f:
    all_querys = pickle.load(f)

with open(query_dir + 'cluster_centroid_dict.pkl', 'rb') as f:
    cluster_centroid_dict = pickle.load(f)

print(len(all_querys))

path_to_database='/Users/lonelu/DesignData/Combs/vdMs/'

### run Search_struct

workdir = '/Users/lonelu/DesignData/ligands_metal_prot/MID1sc10/'

outdir = workdir + 'output_selfcenter/'

target_path = workdir + '5od1_zn.pdb'

win_filter = [('A',35),  ('A', 61),  ('A', 65)]


# workdir = '/mnt/e/DesignData/ligands/LigandBB/_zn_case/6dwv/'

# outdir = workdir + 'output_selfcenter/'

# target_path = workdir + '6dwv.pdb'

# win_filter = [('B', 8), ('B', 10), ('B', 178)]


# workdir = '/mnt/e/DesignData/ligands/LigandBB/8adh/'

# outdir = workdir + 'output_selfcenter/'

# target_path = workdir + '1989_8adh_ZN_1.pdb'

# win_filter = []


# workdir = '/mnt/e/DesignData/ligands/LigandBB/3f7u_lig/'

# outdir = workdir + 'output_selfcenter/'

# target_path = workdir + '3f7u1aa.pdb'

# win_filter = [94, 96, 119]


# workdir = '/mnt/e/DesignData/ligands/LigandBB/2afw_lig/'

# outdir = workdir + 'output_selfcenter/'

# target_path = workdir + '2afw_aa.pdb'

# win_filter = [159, 202, 330]


# workdir = '/mnt/e/DesignData/ligands/LigandBB/huong/'

# outdir = workdir + 'output_selfcenter/'

# target_path = workdir + 'aQ4x_aa.pdb'

# win_filter = ['I-3', 'I-6', 'I-10', 'I-13', 'I-17', 'I-20',
#               'J-3', 'J-6', 'J-7', 'J-10', 'J-13', 'J-14', 'J-17', 'J-20', 'J-21',
#               'K-6', 'K-10', 'K-13', 'K-17', 'K-20',
#               'L-3', 'L-6', 'L-7', 'L-10', 'L-13', 'L-14', 'L-17', 'L-20', 'L-21', 'L-24',
#               'M-3', 'M-6', 'M-10', 'M-13', 'M-17', 'M-20',
#               'N-3', 'N-6', 'N-7', 'N-10', 'N-13', 'N-14', 'N-17', 'N-20', 'N-21'
#             ]

# workdir = '/mnt/e/DesignData/ligands/LigandBB/_reverse_design/c3/c3_vcHHH/'

# outdir = workdir + 'output_selfcenter/'

# target_path = workdir + '5v2g.pdb'

# win_filter = ['A-11',  'B-11',  'C-11']

geometry_path = None
#geometry_path = workdir + 'tetrahydral_geo.pdb'

metal_metal_dist = 0.45

num_contact_vdms = [3]

allowed_aa_combinations = [['H', 'H', 'H']]
#allowed_aa_combinations = []

_filter = filter.Search_filter(filter_abple = False, filter_phipsi = True, max_phipsi_val = 25, 
    filter_vdm_score = False, min_vdm_score = 0, filter_vdm_count = False, min_vdm_clu_num = 20,
    after_search_filter_geometry = True, filter_based_geometry_structure = True, angle_tol = 15, aa_aa_tol = 0.3, aa_metal_tol = 0.2,
    pair_angle_range = [85, 130], pair_aa_aa_dist_range = [2.8, 4], pair_metal_aa_dist_range = None,
    after_search_filter_qt_clash = True, vdm_vdm_clash_dist = 2.7, vdm_bb_clash_dist = 2.2, 
    after_search_open_site_clash = True, open_site_dist = 3.0, 
    write_filtered_result = False, selfcenter_filter_member_phipsi=True)


ss =  search_selfcenter.Search_selfcenter(target_path,  outdir, all_querys, cluster_centroid_dict, query_all_metal, 
    num_contact_vdms, metal_metal_dist, win_filter, validateOriginStruct = True, 
    search_filter= _filter, geometry_path = None, density_radius = 0.6, 
    search_2ndshell = True, secondshell_vdms = path_to_database, rmsd_2ndshell = 1.2,
    allowed_aa_combinations = allowed_aa_combinations, output_wincomb_overlap=True)


#ss.run_selfcenter_search()
search_selfcenter.run_search_selfcenter(ss)

#ss.write_for_combs()

end_time = time.time()
print(end_time - start_time, "seconds")