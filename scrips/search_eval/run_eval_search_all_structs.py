import os
import sys
import prody as pr
import numpy as np
#You can either add the python package path.
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot.search import search, search_eval
from metalprot.basic import filter
import pickle
import multiprocessing as mp

'''
python /mnt/e/GitHub_Design/Metalprot/scrips/search_eval/run_eval_search_all_structs.py
'''
### Please select the correct database based on the eval function you are using. 
#query_dir = '/mnt/e/DesignData/ligands/ZN_rcsb_datesplit/20210624//20210916_2017_2018/pickle_noCys/'
query_dir = '/mnt/e/DesignData/ligands/ZN_rcsb_datesplit/20210624/20210916_2017_2018_selfcenter/pickle_noCYS/'


with open(query_dir + 'AllMetalQuery.pkl', 'rb') as f:
    query_all_metal = pickle.load(f)

with open(query_dir + 'AAMetalPhiPsi.pkl', 'rb') as f:
    all_querys = pickle.load(f)

with open(query_dir + 'cluster_centroid_dict.pkl', 'rb') as f:
    cluster_centroid_dict = pickle.load(f)

with open(query_dir + 'id_cluster_dict.pkl', 'rb') as f:
    id_cluster_dict = pickle.load(f)

cluster_centroid_origin_dict = None 

print(len(all_querys))


# run Search_struct
def run_search(workdir, target_file, query_all_metal, all_querys, cluster_centroid_dict, id_cluster_dict, cluster_centroid_origin_dict):


    outdir = workdir + 'output_eval_' + target_file + '/'

    target_path = workdir + target_file

    print(target_path)

    rmsd_cuts = 0.25

    num_iters = [3]

    win_filter = None

    _filter = filter.Search_filter(filter_abple = False, filter_phipsi = True, max_phipsi_val = 25, 
        filter_vdm_score = False, min_vdm_score = 0, filter_vdm_count = False, min_vdm_clu_num = 20,
        after_search_filter = False, pair_angle_range = [92, 116], pair_aa_aa_dist_range = [3.0, 3.5], pair_metal_aa_dist_range = None,
        filter_qt_clash = False, write_filtered_result = False, selfcenter_filter_member_phipsi = True)

    ss =  search_eval.Search_eval(target_path, outdir, all_querys, id_cluster_dict, cluster_centroid_dict, 
        query_all_metal, cluster_centroid_origin_dict, num_iters, rmsd_cuts, 
        win_filter, validateOriginStruct = True, search_filter= _filter, parallel = False)
    win_search = set()

    try:
        #ss.run_eval_search()
        ss.run_eval_selfcenter_search()
    except:
        return (target_file + ' Error', win_search)

    for k in ss.neighbor_comb_dict.keys():
        win_search.add(k[0])
    
    return (target_file, win_search)


num_cores = int(mp.cpu_count()/2)
pool = mp.Pool(num_cores)

workdir = '/mnt/e/DesignData/ligands/ZN_rcsb_datesplit/20210624/_Seq_core_date_3contact_B45_sub/'

target_files = []
for target_file in os.listdir(workdir):
    if target_file.endswith('.pdb'):
        target_files.append(target_file)

results = [pool.apply_async(run_search, args=(workdir, target_file, query_all_metal, all_querys, cluster_centroid_dict, id_cluster_dict, cluster_centroid_origin_dict)) for target_file in target_files]
results = [p.get() for p in results]

with open(workdir + '_summary.txt', 'w') as f:
    f.write('target_file\twin_extract\n')
    for r in results:
        try:
            f.write(r[0] + '\t')
            f.write(str(r[1]) + '\t')
        except:
            f.write(r[0] + '\t\n')

