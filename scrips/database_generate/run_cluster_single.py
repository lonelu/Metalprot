import os
import sys
import prody as pr
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot.database import database_extract as ldb
from metalprot.database import database_cluster as ldb_clu

def extract_single_vdm(cores, outdir, AA, key, basic = True, extention = None, n = None, key_out = None, phipsi = False):
    for c in cores:   
        if basic:
            c.generate_AA_Metal(AA, key)
        elif extention:
            c.generate_AA_ext_Metal(AA, extention, key)          
        elif n:
            c.generate_AA_kMetal(AA, n, key, key_out) 
        elif phipsi:
            c.generate_AA_phipsi_Metal(AA, key)
        c.write_vdM(outdir, key = key)
    return

### set up parameters

workdir = "/mnt/e/DesignData/ligands/Zn_rcsb_datesplit/20210624/"

metal_sel = 'ion or name NI MN ZN CO CU MG FE' 
align_sel_backbone = 'name C CA N O NI MN ZN CO CU MG FE or ion'

cores = ldb.load_cores(workdir + '_Seq_core_date_reps/')


AA = 'CYS'

len_sel = 9

#Change len_sel according to aa sidechain number. His:6, Glu: 5, Asp: 4, Cys: 2
if AA == 'HIS':
    len_sel_sc = len_sel + 6 
elif AA == 'GLU':
    len_sel_sc = len_sel + 5
elif AA == 'ASP':
    len_sel_sc = len_sel + 4 
elif AA == 'CYS':
    len_sel_sc = len_sel + 2 

        
### Align aa core w/o sc------------------------------------------------------------------------------------
'''
# AAMetal_HIS
extract_single_vdm(cores, workdir + 'M1_AAMetal_' + AA + '_reps/', AA = AA, key = 'AAMetal_HIS', basic = True)

_pdbs = ldb.get_all_pbd_prody(workdir + 'M1_AAMetal_' + AA + '_reps/')

ldb.run_cluster(_pdbs, workdir, 'M1_AAMetal_' + AA + '_cluster02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = len_sel, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm1_')

ldb.run_cluster(_pdbs, workdir, 'M1-1_AAMetalSc_' + AA + '_cluster02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = len_sel_sc, align_sel = 'heavy', min_cluster_size = 2, tag = 'm1-1_') 

#ldb.run_cluster(_pdbs, workdir, 'M1_AAMetalTest_HIS_cluster05/', rmsd = 0.5, metal_sel = metal_sel, len_sel = 5, align_sel = 'name C CA N CB ZN', min_cluster_size = 2, tag = 'm1-1_') 
'''

'''
# AA5Metal_HIS
extract_single_vdm(cores, workdir + "M1_AA5Metal_HIS_reps/", AA = AA, key = 'AAMetal_HIS', basic = False, n = 5, key_out='AA5Metal_HIS')

_pdbs2 = ldb.get_all_pbd_prody(workdir + "M1_AA5Metal_HIS_reps/")

ldb.run_cluster(_pdbs2, workdir, 'M1_AA5Metal_HIS_cluster02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 9, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm1_')

ldb.run_cluster(_pdbs2, workdir, 'M1_AA5MetalSc_HIS_cluster02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 15, align_sel = 'heavy', min_cluster_size = 2, tag = 'm1-1_') 
'''

### Align his core +- 1 AA------------------------------------------------------------------------------------ 
'''
# AAExtMetal_HIS_ext1
extract_single_vdm(cores, workdir + 'M3-1_AAextMetal_' + AA + '_reps/', AA = AA, key = 'AAextMetal_' + AA + '_ext3', basic = False, extention=1, n = None, key_out=None)

_pdbs = ldb.get_all_pbd_prody(workdir + 'M3-1_AAextMetal_' + AA + '_reps/')

ldb.run_cluster(_pdbs, workdir, 'M3-1_AAextMetal_' + AA + '_cluster02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 13, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm3-1_')
'''


'''
# AAExt5Metal_HIS_ext1
extract_single_vdm(cores, workdir + "M3-1_AAext5Metal_' + AA + '_reps/", AA = AA, key = 'AAextMetal_' + AA + '_ext3', extention=1, n = 5, key_out='AAext5Metal_' + AA + '_ext3')

_pdbs = ldb.get_all_pbd_prody(workdir + 'M3-1_AAext5Metal_' + AA + '_reps/')

ldb.run_cluster(_pdbs, workdir, 'M3-1_AAext5Metal_' + AA + '_cluster02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 17, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm3-1_')
'''

### Align his core with phipsi------------------------------------------------------------------------------------ 

# AAMetal_HIS
outdir = workdir + '20210916_2017_2018/'
os.makedirs(outdir, exist_ok = True)

extract_single_vdm(cores, outdir + 'AAMetalPhiPsi_' + AA + '_reps/', AA = AA, key = 'AAMetalPhiPsi_' + AA, basic = False, phipsi=True)

_pdbs = ldb.get_all_pbd_prody(outdir + 'AAMetalPhiPsi_' + AA + '_reps/')

ldb_clu.run_cluster(_pdbs, outdir, 'AAMetalPhiPsi_' + AA + '_cluster05/', rmsd = 0.5, metal_sel = metal_sel, len_sel = len_sel_sc, align_sel = 'heavy', min_cluster_size = 0, tag = 'AAMetalPhiPsi_' + AA + '_')

