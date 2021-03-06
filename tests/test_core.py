from metalprot.apps.core import Core
import pytest
import os
import shutil
import prody as pr

from metalprot import ligand_database
from metalprot import core

def test_core_kMetal():

    #workdir = '/mnt/e/GitHub_Design/Metalprot/tests/test_data/'
    workdir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'

    pdb_prody = pr.parsePDB(workdir + '5od1_zn.pdb')

    core = Core(pdb_prody)

    core.generate_AA_Metal(AA = 'HIS')

    core.generate_AA_kMetal(AA = 'HIS', k = 5, key = 'AA_Metal', key_out = 'AA_kMetal')

    core.write_vdM(workdir + 'output/', key = 'AA_kMetal')

    pdb_kMetal = pr.parsePDB(workdir + 'output/' + '5od1_zn_AA_kMetal_mem0.pdb')

    assert len(pdb_kMetal.select('name ZN')) == 5

    #shutil.rmtree(workdir + 'output/')

def test_core_binary_contact():

    workdir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'

    pdb_prody = pr.parsePDB(workdir + '5od1_zn.pdb')

    core = Core(pdb_prody)

    key = 'BinaryAtomContact'

    core.generate_binary_contact(key)

    core.write_vdM(workdir + 'output/', key)

    assert len(core.atomGroupDict[key]) == 3

    assert len(core.atomGroupDict[key][0]) == 3

    core.generate_atom_contact('AtomContact')

    assert len(core.atomGroupDict['AtomContact4'][0]) == 4


def test_contact():
    workdir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'

    pdbs = [None]*3
    pdbs[0] = pr.parsePDB(workdir + '5_1_3.37_m1-1_cluster_18_mem_20_5od1_ZN_1_AAMetal_HIS_mem0.pdb')
    pdbs[1] = pr.parsePDB(workdir + '5_2_3.37_m1-1_cluster_20_mem_20_4k1r_ZN_5_AAMetal_HIS_mem0.pdb')
    pdbs[2] = pr.parsePDB(workdir + '5_3_3.37_m1-1_cluster_22_mem_31_5od1_ZN_1_AAMetal_HIS_mem1.pdb')

    contact_pdb = core.get_contact(pdbs)

    assert len(contact_pdb) == 4
    assert contact_pdb.getNames()[0] == 'ND1'
    assert contact_pdb.getNames()[-1] == 'ZN'

def test_2ndshell():

    workdir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'

    pdb_prody = pr.parsePDB(workdir + '5od1_ZN_1_2nd.pdb')

    core = Core(pdb_prody)

    key = '_2ndShell'

    core.generate_AA_2ndShell_Metal(key, filter_AA=True, AA='HIS')

    core.write_vdM(workdir + 'output/', key)

    assert len(core.atomGroupDict[key]) == 1


    resind = core.contact_aa_resinds[2]
    _2nshell_resinds = ligand_database.get_2ndshell_indices([resind], pdb_prody, pdb_prody.select('name ZN')[0].getIndex())
    assert len(_2nshell_resinds) == 0

