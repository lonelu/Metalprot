import os
import sys
import prody as pr
import numpy as np
#You can either add the python package path.
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot import search_struct, extract_vdm, ligand_database
from metalprot import hull
'''
python /mnt/e/GitHub_Design/Metalprot/metalprot/scrips/run_search_struct_3vdm.py
'''

### Generate queryss
queryss = []

query_dir = '/mnt/e/DesignData/ligands/ZN_rcsb_datesplit/20210624/'

# Get query pdbs, Add the cluster metal points into the query.hull_points

querys = extract_vdm.extract_all_centroid(query_dir, summary_name = '_summary.txt', file_name_includes = ['AAMetalPhiPsi_HIS'], score_cut = 0, clu_num_cut = 50)


for query in querys:
    clu = extract_vdm.get_vdm_cluster(query)
    clu.realign_by_CCAN(target = query)
    query.cluster = clu
    query.extract_mem_metal_point()

""" align_sel = 'name N CA C'
query_all_metal = querys[0].copy()
clu_rank = 0
for query in querys:
    inds = np.unique(query.query.getResindices())
    ind = inds[int(inds.shape[0]/2)-1]
    transform = pr.calcTransformation(query.query.select(align_sel + ' and resindex ' + str(ind)), query_all_metal.query.select(align_sel + ' and resindex ' + str(query_all_metal.contact_resind)))
    transform.apply(query.query)

    clu = extract_vdm.get_vdm_cluster(query)
    clu.realign_by_CCAN(target = query, align_sel=align_sel)
    query.cluster = clu
    query.extract_mem_metal_point()

    # for c in query.cluster.querys:
    #     outdir = query_dir + 'AAMetalPhiPsi_HIS_realgin/' + str(clu_rank) + '/'
    #     if not os.path.exists(outdir):
    #         os.mkdir(outdir)
    #     pr.writePDB(outdir + c.query.getTitle(), c.query)
    # clu_rank +=1

#-----------------------------------------
align_sel = 'name N CA C'
query_all_metal = querys[0].copy()
points = []
count = 0
for _query in querys:
    query = _query.copy()
    inds = np.unique(query.query.getResindices())
    ind = inds[int(inds.shape[0]/2)-1]
    #transform = pr.calcTransformation(query.query.select(align_sel + ' and resindex ' + str(ind)), query_all_metal.query.select(align_sel + ' and resindex ' + str(query_all_metal.contact_resind)))
    #transform.apply(query.query)
    #transform.apply(query.hull_ag)
    points.extend(query.hull_ag.getCoords())
    #query_all_metal.hull_ag = hull.transfer2pdb(points)
    hull.write2pymol(points, query_dir, str(count) + '_' + query_all_metal.query.getTitle() + '_original.pdb')
    count += 1 """

print(len(querys))

queryss.append(querys)

#contact_querys = extract_vdm.extract_all_centroid(query_dir, summary_name = '_summary.txt', file_name_includes = ['M8_AtomContact4_clusters'], score_cut = 0, clu_num_cut = 2)
contact_querys = None
#_2nd_querys = extract_vdm.extract_all_centroid(query_dir + '20210608/', summary_name = '_summary.txt', file_name_includes = ['M7_AA2sMetal-HIS_clusters'], score_cut = 0, clu_num_cut = 0)

_2nd_querys = None

### run Search_struct

workdir = '/mnt/e/DesignData/ligands/LigandBB/MID1sc10/'

outdir = workdir + 'output_hull_all_TestWriteSummary/'

target_path = workdir + '5od1_zn.pdb'

rmsd_cuts = [0.5, 0.5, 0.5]

dist_cuts = [1, 1, 1]

num_iter = 3

clash_query_query = 2.3

clash_query_target = 2.3

use_sep_aas = [False, False, False]

tolerance = 0.5

fine_dist_cut = 0.2

win_filter = None

'''
#For testing natural metal binding proteins.
_pdb = pr.parsePDB(target_path)

cores = ligand_database.get_metal_core_seq(_pdb, metal_sel = 'ion or name NI MN ZN CO CU MG FE' , extend = 4)

print(cores[0][1].select('name CA').getSequence())

#pr.writePDB(target_path + '_core.pdb', cores[0][1])

win_filter = [x for x in np.unique(cores[0][1].getResindices())][0:-1]

print(win_filter)
'''

#win_filter = [30, 31, 32, 33, 34, 35, 36, 37, 38, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]
#win_filter = [34,  60,  64]

ss = search_struct.Search_struct(target_path, outdir, queryss, rmsd_cuts, dist_cuts, num_iter, clash_query_query, clash_query_target, use_sep_aas, 
    tolerance, fine_dist_cut = fine_dist_cut, win_filter = win_filter, contact_querys = contact_querys, secondshell_querys=_2nd_querys, validateOriginStruct=False, 
    query_all_metal = None)

#cquery = supperimpose_target_bb(query, ss.target, [34], ss.rmsd_cuts[0], query_align_sel='name N CA C and resindex ' + str(query.contact_resind), target_align_sel='name N CA C', validateOriginStruct = ss.validateOriginStruct)

#ss.run_hull_based_search()
ss.hull_generate_query_dict()
ss.hull_generate_pairwise_win_dict()
ss.hull_iter_win()
#ss.hull_win2indcomb((34, 60, 64))
ss.hull_construct()
ss.hull_calc_comb_score()
ss.hull_calc_geometry()

ss.hull_write()
ss.hull_write_summary()

#ss.run_search_2nshells(outpath = '/mem_combs/', rmsd=0.5)
### If only search 2nshell for a specific comb.
#ss.search_2ndshell(4)
#ss.write_2ndshell(4)

