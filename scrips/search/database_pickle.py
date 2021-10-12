'''
Here we load all vdm database into pkl file. pkl file could be load fast.

The database should be categoried.

all_vdms = [vdm]
all_metal_vdm = vdm with all member metal position.
centroid_query_dict = [clu_key:vdm id] for example '(HIS, 1234): 4567' 

'''

import os
import sys
import prody as pr
import numpy as np
#You can either add the python package path.
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot.search import extract_vdm
from metalprot.basic import hull, quco
import pickle

'''
python /mnt/e/GitHub_Design/Metalprot/scrips/search/database_pickle.py
'''

query_dir = '/mnt/e/DesignData/ligands/ZN_rcsb_datesplit/20211008/20211009_category/'

# Get query pdbs, Add the cluster metal points into the query.hull_points

centroid_querys = extract_vdm.extract_all_centroid(query_dir, summary_name = '_summary.txt', file_name_includes = ['AAMetalPhiPsi', 'cluster'], file_name_not_includes= ['CYS'], score_cut = 0, clu_num_cut = 0)

#Prepare all the data, this should be optimized in the future.

metal_sel = 'ion or name NI MN ZN CO CU MG FE' 
align_sel = 'name N CA C'

all_metal_vdm = centroid_querys[0].copy()
all_vdms = []

all_coords = []
cluster_centroid_dict = {}

query_id = 0
for _query in centroid_querys:
    query = _query.copy()
    ind = query.contact_resind
    transform = pr.calcTransformation(query.query.select(align_sel + ' and resindex ' + str(ind)), all_metal_vdm.query.select(align_sel + ' and resindex ' + str(all_metal_vdm.contact_resind)))
    transform.apply(query.query)

    mem_vdms = extract_vdm.get_mem_vdms(query)
    clu = quco.Cluster(mem_vdms)
    #clu.realign_by_HEAVY_candidates(target = query, align_sel='heavy') #The position of each point is decided by how the vdM is supperimposed.
    clu.realign_by_CCAN(target = query, align_sel=align_sel)
    for q in clu.querys:
        q.id = query_id
        all_vdms.append(q)
        if 'centroid' in q.query.getTitle():
            q.clu_rank = int(q.query.getTitle().split('_')[3])
            cluster_key = q.get_cluster_key()
            cluster_centroid_dict[cluster_key] = query.id
        all_coords.append(q.query.select(metal_sel)[0].getCoords())
        query_id += 1

    query.clu_rank = int(query.query.getTitle().split('_')[3]) 



all_metal_vdm.metal_atomgroup = hull.transfer2pdb(all_coords)

outdir = query_dir + 'pickle_noCYS/'
os.makedirs(outdir, exist_ok= True)

with open(outdir + 'all_metal_vdm.pkl', 'wb') as f:
    pickle.dump(all_metal_vdm, f)

with open(outdir + 'AAMetalPhiPsi.pkl', 'wb') as f:
    pickle.dump(all_vdms, f)

with open(outdir + 'cluster_centroid_dict.pkl', 'wb') as f:
    pickle.dump(cluster_centroid_dict, f)


'''
# depre function
cluster_centroid_origin_dict = {}
for _query in centroid_querys:
    query = _query.copy()
    cluster_key = query.get_cluster_key()
    cluster_centroid_origin_dict[cluster_key] = query
    cluster_coords = []

    clu = extract_vdm.get_vdm_cluster(query)
    for q in clu.querys:
        cluster_coords.append(q.query.select(metal_sel)[0].getCoords())
    cluster_centroid_origin_dict[cluster_key].hull_ag = hull.transfer2pdb(cluster_coords)

with open(outdir + 'cluster_centroid_origin_dict.pkl', 'wb') as f:
    pickle.dump(cluster_centroid_origin_dict, f)
'''