import prody as pr
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors


def rotate_ligs(orign_lig, rot, rest, rotation_degree = 10, dist = 3):
    '''
    Note that the ZN must be the last element of the ligand.
    '''
    lig = orign_lig.copy()

    # transform the lig to z axis.
    rot_coords = lig.select('name ' + ' '.join(rot)).getCoords()
    rot_dist = pr.calcDistance(rot_coords[1], rot_coords[0])

    z_coords = np.zeros((2, 3), dtype=float)
    z_coords[1, -1] = rot_dist

    pr.calcTransformation(rot_coords, z_coords).apply(lig)

    all_ligs = []
    for i in range(0, 360, rotation_degree):
        _lig = lig.copy()
        rotation = Rotation.from_rotvec(np.radians(i)*np.array([0, 0, 1]))        
        _coords = rotation.apply(_lig.getCoords())
        _lig.setCoords(_coords)
        if len(rest) > 0:
            _lig.select('name ' + ' '.join(rest)).setCoords(lig.select('name ' + ' '.join(rest)).getCoords())

        pr.calcTransformation(_lig.select('name ' + ' '.join(rest)).getCoords(), orign_lig.select('name ' + ' '.join(rest)).getCoords()).apply(_lig)
        _lig.setTitle(lig.getTitle() + '_' + '-'.join(rot) + '_' + str(i))
        #pr.writePDB(workdir + 'ligand_rotation/' +_lig.getTitle() + '_' + '-'.join(rot) + '_' + str(i), _lig)

        if ligand_rot_is_clash(_lig, rot, rest, dist):
            continue
        all_ligs.append(_lig)

    return all_ligs


def generate_rotated_ligs(lig, rots, rests, rotation_degrees, clash_dist = 3):
    '''
    The method only works for 2 rots.
    TO DO: use recursive algorithm to allow more than 2 rots.
    '''
    all_ligs = []
    i = 0
    rot = rots[i]
    rest = rests[i]
    degree = rotation_degrees[i]
    ligs = rotate_ligs(lig, rot, rest, degree, clash_dist)
    for _lig in ligs:
        i = 1
        rot = rots[i]
        rest = rests[i]
        degree = rotation_degrees[i]
        ligs2 = rotate_ligs(_lig, rot, rest, degree, clash_dist)
        all_ligs.extend(ligs2)
    return all_ligs


def ligand_rot_is_clash(lig, rot, rest, dist = 3):
    '''
    The ligand it self could clash after rotation.
    The idea is to sep the ligand by rot into 2 groups anc check their dists.

    return True if clash
    '''
    atoms_rot = lig.select('heavy and not name ' + ' '.join(rot) + ' ' + ' '.join(rest)).getCoords()
    rest_coords = lig.select('heavy and name ' +  ' '.join(rest)).getCoords()

    nbrs = NearestNeighbors(radius= dist).fit(atoms_rot)
    adj_matrix = nbrs.radius_neighbors_graph(rest_coords).astype(bool)

    if np.sum(adj_matrix) >0:
        return True
    return False


def lig_2_ideageo(ligs, lig_connects, ideal_geo_o = None, geo_sel = 'name OE2 ZN', metal_sel = 'name NI MN ZN CO CU MG FE'):
    '''
    supperimpose the ligand to the ideal metal binding geometry.
    '''
    _lig = ligs[0]

    lig_sel = _lig.select('name ' + ' '.join(lig_connects))

    ideal_geo_sel = ideal_geo_o.select(geo_sel)

    transformation = pr.calcTransformation(lig_sel, ideal_geo_sel)

    for lg in ligs:
        transformation.apply(lg)

    return 


def ligand_clashing_filter(ligs, target, dist = 3):
    '''
    The ligand clashing: the ligs cannot have any heavy atom within 3 A of a target bb.
    Nearest neighbor is used to calc the distances. 
    '''
    all_coords = []
    labels = []

    for i in range(len(ligs)):
        coords = ligs[i].select('heavy').getCoords()
        all_coords.extend(coords)
        labels.extend([i for j in range(len(coords))])

    
    target_coords = target.select('name N C CA O CB').getCoords()

    nbrs = NearestNeighbors(radius= dist).fit(target_coords)
    adj_matrix = nbrs.radius_neighbors_graph(all_coords).astype(bool)

    failed = set()
    for i in range(adj_matrix.shape[0]):
        if adj_matrix.getrow(i).toarray().any():
            failed.add(labels[i])
    
    filtered_ligs = []
    for i in range(len(ligs)):
        if i in failed:
            continue
        filtered_ligs.append(ligs[i])

    return filtered_ligs
    


