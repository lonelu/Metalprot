from metalprot.search import search_selfcenter
from metalprot.basic import filter
import metalprot.basic.constant as constant
import pickle
import time
import prody as pr


'''
python /mnt/e/GitHub_Design/Metalprot/scrips/search_selfcenter/run_selfcenter_search.py

'''
start_time = time.time()

query_dir = '/mnt/e/DesignData/ligands/ZN_rcsb_datesplit/20211013/20211013_selfcenter/pickle_all/'

with open(query_dir + 'all_metal_vdm.pkl', 'rb') as f:
    query_all_metal = pickle.load(f)

with open(query_dir + 'AAMetalPhiPsi.pkl', 'rb') as f:
    all_querys = pickle.load(f)

with open(query_dir + 'cluster_centroid_dict.pkl', 'rb') as f:
    cluster_centroid_dict = pickle.load(f)

print(len(all_querys))


### run Search_struct

workdir = '/mnt/e/DesignData/ligands/LigandBB/MID1sc10/'

outdir = workdir + 'output_selfcenter/'

target_path = workdir + '5od1_zn.pdb'

#win_filter = []
win_filter = [35,  61,  65]



metal_metal_dist = 0.45

num_contact_vdms = [3]

allowed_aa_combinations = [['H', 'H', 'H']]

_filter = filter.Search_filter(filter_abple = False, filter_phipsi = True, max_phipsi_val = 25, 
    filter_vdm_score = False, min_vdm_score = 0, filter_vdm_count = False, min_vdm_clu_num = 20,
    after_search_filter = True, pair_angle_range = [92, 116], pair_aa_aa_dist_range = [3.0, 3.5], pair_metal_aa_dist_range = None,
    filter_qt_clash = True, write_filtered_result = False, selfcenter_filter_member_phipsi=True)

ss =  search_selfcenter.Search_selfcenter(target_path, outdir, all_querys, cluster_centroid_dict, query_all_metal, 
    num_contact_vdms, metal_metal_dist, win_filter, validateOriginStruct = True, search_filter= _filter, density_radius = 0.6,
    allowed_aa_combinations = allowed_aa_combinations)

search_selfcenter.run_search_selfcenter(ss)

end_time = time.time()
print(end_time - start_time, "seconds")


#ligand positioning.

lig_path = workdir + 'Zn_Phenol.pdb'

lig = pr.parsePDB(lig_path)

lig_connects = ['OH', 'ZN']

ro1 = ['ZN', 'OH']
ro2 = ['OH', 'CZ']

rotation_degree = 5

key = list(ss.best_aa_comb_dict.keys())[0]


ideal_geo = ss.best_aa_comb_dict[key].ideal_geo

ideal_geo_o = constant.tetrahydra_geo_o
pr.calcTransformation(ideal_geo_o.select('not oxygen'), ideal_geo).apply(ideal_geo_o)
pr.writePDB(workdir + ideal_geo_o.getTitle(), ideal_geo_o)

all_ligs = generate_rotated_ligs(lig, ro1, ro2, rotation_degree = 45)
# for l in all_ligs: 
#     pr.writePDB(workdir + 'ligand_rotation/' +  l.getTitle(), l)

tf = calc_lig2ideageo_transformation(all_ligs[0], lig_connects, ideal_geo_o)


for lg in all_ligs:
    tf.apply(lg)

pr.writePDB(workdir + lg.getTitle(), lg)