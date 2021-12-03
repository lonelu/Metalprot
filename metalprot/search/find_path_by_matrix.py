import os
import numpy as np
import itertools
from numpy.core.shape_base import _block_slicing
import prody as pr
import datetime
from scipy.sparse.csr import csr_matrix
from scipy.sparse import lil_matrix

from scipy.sparse.sputils import matrix

from ..basic import hull
from ..basic import utils
from ..basic.filter import Search_filter
from ..basic.constant import one_letter_code
from .graph import Graph
from .comb_info import CombInfo

from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

'''
test_matrix = np.array(
    [[0, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]],
    dtype = bool
)

test_paths = calc_adj_matrix_paths(test_matrix, num_iter)

paths = calc_adj_matrix_paths(m_adj_matrix)

for i in range(len(ss.vdms)):
    if '5od1' in ss.vdms[i].query.getTitle():
        print(i)
        print(ss.vdms[i].query.getTitle())
index = [3776, 4387*1+2865, 4387*2+2192]
index == [3776, 7252, 10966]

all_vdms = []
for i in range(len(wins)):
    all_vdms.extend(ss.vdms)
all_vdms[7252].query.getTitle()


adj_matrix[3776, 7252]
adj_matrix[3776, 10966]
adj_matrix[7252, 10966]

win_mask[3776, 7252]
win_mask[3776, 10966]
win_mask[7252, 10966]


paths = calc_adj_matrix_paths(m_adj_matrix)

'''

def calc_adj_matrix_paths(m_adj_matrix, num_iter =3):
    paths = []
    for r in range(m_adj_matrix.shape[0]):
        inds = m_adj_matrix.rows[r]
        if len(inds) < num_iter-1:
            continue
        for _comb in itertools.combinations(inds, num_iter-1):
            comb = [r]
            comb.extend(_comb)
            valid = calc_adj_matrix_paths_helper(m_adj_matrix, comb, num_iter, 1)
            if valid:
                #print(comb)
                paths.append(comb)

    return paths


def calc_adj_matrix_paths_helper( m_adj_matrix, comb, num_iter, iter):
    if iter >= num_iter -1:
        return True
    r_curr = comb[iter]
    for r_next in comb[iter+1:]:
        if not m_adj_matrix[r_curr, r_next]:
            return False
    return calc_adj_matrix_paths_helper(m_adj_matrix, comb, num_iter, iter +1)


def neighbor_generate_nngraph(ss):
    '''
    Instead of doing this in a pairwise way as the function 'search.neighbor_generate_pair_dict'.
    Here we calc one nearest neighbor object and graph.

    ss is the search.Search_vdM object.
    '''
    wins = sorted(list(ss.neighbor_query_dict.keys()))
    metal_vdm_size = len(ss.all_metal_vdm.get_metal_mem_coords())

    #calc radius_neighbors_graph
    all_coords = []
    win_labels = []
    vdm_inds = []
    for inx in range(len(wins)):
        wx = wins[inx] 
        win_labels.extend([wx]*metal_vdm_size)
        n_x = ss.neighbor_query_dict[wx].get_metal_mem_coords()
        all_coords.extend(n_x)
        vdm_inds.extend(range(metal_vdm_size))

    #nbrs = NearestNeighbors(radius= ss.metal_metal_dist).fit(all_coords)
    #adj_matrix = nbrs.radius_neighbors_graph(all_coords).astype(bool)
    adj_matrix = radius_neighbors_graph(all_coords, radius= ss.metal_metal_dist).astype(bool)
    #print(adj_matrix.shape)
    #metal_clashing with bb
    bb_coords = ss.target.select('name N C CA O').getCoords()
    nbrs_bb = NearestNeighbors(radius= 3.5).fit(all_coords)
    adj_matrix_bb = nbrs_bb.radius_neighbors_graph(bb_coords).astype(bool)
    #print(adj_matrix_bb.shape)

    # #create mask
    # mask = generate_filter_mask(ss, wins, win_labels, metal_vdm_size, adj_matrix_bb)
    # #calc modified adj matrix
    # m_adj_matrix = adj_matrix.multiply(mask)

    m_adj_matrix = filter_adj_matrix(ss, wins, metal_vdm_size, adj_matrix, adj_matrix_bb)

    return m_adj_matrix, win_labels, vdm_inds


def generate_filter_mask(ss, wins, win_labels, metal_vdm_size, adj_matrix_bb):
    '''
    One issue for the mask method is that the matrix is sparse, there will be a lot of unnecessary calculation.
    For example, filter phi psi, if a vdm is never be in any neighbor, there is no need to calc it.
    '''
        
    wins = sorted(list(ss.neighbor_query_dict.keys()))
    metal_vdm_size = len(ss.all_metal_vdm.get_metal_mem_coords())    
    
    win_mask = np.ones((len(win_labels), len(win_labels)), dtype=bool)
    win_mask = np.triu(win_mask)

    # win filter: vdm on the same position don't connect.
    for inx in range(len(wins)):
        win_mask[inx*metal_vdm_size:(inx+1)*metal_vdm_size, inx*metal_vdm_size:(inx+1)*metal_vdm_size] = 0

    # Metal bb Clashing filter.
    metal_clashing_vec = np.ones(len(win_labels), dtype=bool)
    for i in range(adj_matrix_bb.shape[0]):
        metal_clashing_vec *= ~(adj_matrix_bb[i].toarray().reshape(len(win_labels),))
    labels_m = np.broadcast_to(metal_clashing_vec, (len(wins)*metal_vdm_size, len(wins)*metal_vdm_size))
    win_mask *= labels_m.T
    win_mask *= labels_m
    

    # Modify mask with aa filter. 
    if ss.validateOriginStruct:
        v_aa = np.array([v.aa_type for v in ss.vdms])
        ress = [one_letter_code[ss.target.select('name CA and resindex ' + str(wx)).getResnames()[0]] for wx in wins]
        labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = v_aa == ress[inx]
        labels_m = np.broadcast_to(labels, (len(wins)*metal_vdm_size, len(wins)*metal_vdm_size))
        win_mask *= labels_m.T
        win_mask *= labels_m

    if ss.search_filter.filter_abple:
        v_abples = np.array([v.abple for v in ss.vdms])
        apxs = [ss.target_abple[wx] for wx in wins]
        labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = v_abples == apxs[inx]
        labels_m = np.broadcast_to(labels, (len(wins)*metal_vdm_size, len(wins)*metal_vdm_size))
        win_mask *= labels_m.T
        win_mask *= labels_m      

    #Filter unwanted amino acids. if ss.allowed_aas = {'H', 'D'}, then {'E', 'S'} will be eliminated.
    if not ss.validateOriginStruct and len(ss.allowed_aas) > 0 and len(ss.allowed_aas) < 4:
        v_aa = np.array([v.aa_type for v in ss.vdms])
        labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = np.array([v in ss.allowed_aas for v in v_aa])
        labels_m = np.broadcast_to(labels, (len(wins)*metal_vdm_size, len(wins)*metal_vdm_size))
        win_mask *= labels_m.T
        win_mask *= labels_m 

    if ss.search_filter.filter_phipsi:
        #TO DO: filter phi psi need to be changed to be able to broadcast.
        v_phis = [v.phi for v in ss.vdms]
        v_psis = [v.psi for v in ss.vdms]

        phis = [ss.phipsi[wx][0] for wx in wins]
        psis = [ss.phipsi[wx][1] for wx in wins]
        labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):  
            for i in range(metal_vdm_size):
                #TO DO: how to ignore unnecessary psiphi
                #if any(win_mask[inx*metal_vdm_size + i, ]) and any(adj_matrix[inx*metal_vdm_size + i, ]):
                phi_ok = utils.filter_phipsi(phis[inx], v_phis[i], ss.search_filter.max_phipsi_val)
                psi_ok = utils.filter_phipsi(psis[inx], v_psis[i], ss.search_filter.max_phipsi_val)
                if phi_ok and psi_ok:
                    labels[inx*metal_vdm_size + i] = True
        labels_m = np.broadcast_to(labels, (len(wins)*metal_vdm_size, len(wins)*metal_vdm_size))
        win_mask *= labels_m.T
        win_mask *= labels_m  
            
    return win_mask


def filter_adj_matrix(ss, wins, metal_vdm_size, adj_matrix, adj_matrix_bb):
    '''
    One issue for the mask method is that the matrix is sparse, there will be a lot of unnecessary calculation.
    For example, filter phi psi, if a vdm is never be in any neighbor, there is no need to calc it.
    '''
        
    wins = sorted(list(ss.neighbor_query_dict.keys()))
    metal_vdm_size = len(ss.all_metal_vdm.get_metal_mem_coords())    
    
    mask_labels = np.ones(len(wins)*metal_vdm_size, dtype=bool)

    # Metal bb Clashing filter
    for i in range(adj_matrix_bb.shape[0]):
        mask_labels *= ~(adj_matrix_bb[i].toarray().reshape(len(mask_labels),))


    #aa origin filter. 
    if ss.validateOriginStruct:
        v_aa = np.array([v.aa_type for v in ss.vdms])
        ress = [one_letter_code[ss.target.select('name CA and resindex ' + str(wx)).getResnames()[0]] for wx in wins]
        aa_labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            aa_labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = v_aa == ress[inx]
        mask_labels *= aa_labels

    # filter vdM by score or count:
    if ss.search_filter.filter_vdm_score:
        v_scores = np.array([v.score for v in ss.vdms])
        vdm_score_labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            vdm_score_labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = v_scores >= ss.search_filter.min_vdm_score
        mask_labels *= vdm_score_labels

    if ss.search_filter.filter_vdm_score:
        v_count = np.array([v.score for v in ss.vdms])
        vdm_count_labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            vdm_count_labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = v_count >= ss.search_filter.min_vdm_clu_num
        mask_labels *= vdm_count_labels

    # abple filter
    if ss.search_filter.filter_abple:
        v_abples = np.array([v.abple for v in ss.vdms])
        apxs = [ss.target_abple[wx] for wx in wins]
        abple_labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            abple_labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = v_abples == apxs[inx]
        mask_labels *= abple_labels

    #Filter unwanted amino acids. if ss.allowed_aas = {'H', 'D'}, then {'E', 'S'} will be eliminated.
    if not ss.validateOriginStruct and len(ss.allowed_aas) > 0 and len(ss.allowed_aas) < 4:
        v_aa = np.array([v.aa_type for v in ss.vdms])
        aa_allow_labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):
            aa_allow_labels[inx*metal_vdm_size:(inx+1)*metal_vdm_size] = np.array([v in ss.allowed_aas for v in v_aa])
        mask_labels *= aa_allow_labels

    #phi psi filter
    if ss.search_filter.filter_phipsi:
        #TO DO: filter phi psi need to be changed to be able to broadcast.
        v_phis = [v.phi for v in ss.vdms]
        v_psis = [v.psi for v in ss.vdms]

        phis = [ss.phipsi[wx][0] for wx in wins]
        psis = [ss.phipsi[wx][1] for wx in wins]
        phipsi_labels = np.zeros(len(wins)*metal_vdm_size, dtype=bool)
        for inx in range(len(wins)):  
            for i in range(metal_vdm_size):
                phi_ok = utils.filter_phipsi(phis[inx], v_phis[i], ss.search_filter.max_phipsi_val)
                psi_ok = utils.filter_phipsi(psis[inx], v_psis[i], ss.search_filter.max_phipsi_val)
                if phi_ok and psi_ok:
                    phipsi_labels[inx*metal_vdm_size + i] = True
        mask_labels *= phipsi_labels

    m_adj_matrix = lil_matrix(adj_matrix.shape, dtype=bool)
    for r in range(adj_matrix.shape[0]):
        if not mask_labels[r]:
            continue 

        for ind in range(adj_matrix.indptr[r], adj_matrix.indptr[r+1]):
            c = adj_matrix.indices[ind]
            
            # vdm on the same position don't connect.
            if c > r and mask_labels[c] and r//metal_vdm_size != c//metal_vdm_size:
                m_adj_matrix[r, c] = True
            
    return m_adj_matrix