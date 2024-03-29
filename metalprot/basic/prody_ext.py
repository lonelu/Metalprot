'''
To help manipulate pdb with prody functions.
'''

import prody as pr
import string
import numpy as np
from . import constant


def transfer2resindices(pdb, chidress):
    '''
    Input of [resnum] or [chid-resnum] or [seg-chid-resnum] transfer to [resindex]
    '''
    target_resnums = pdb.select('protein and name CA').getResnums()
    target_chids = pdb.select('protein and name CA').getChids()
    ind2chidres = {}
    chidres2ind = {}
    for ind in range(len(target_resnums)):
        # if len(np.unique(target_chids)) != 1:
        #     chid_resnum = target_chids[ind] + '-' + str(target_resnums[ind])
        # else:
        #     chid_resnum = str(target_resnums[ind])
        chid_resnum = (target_chids[ind], target_resnums[ind])
        chidres2ind[chid_resnum] = ind
        ind2chidres[ind] = chid_resnum
    resindices = [chidres2ind[x] for x in chidress]
    return resindices, chidres2ind, ind2chidres

def transfer2pdb(points, names = None, resnums = None, resnames = None, title = 'MetalMol'):
    '''
    Speically used for Metal binding geometry pdb. 
    '''
    if not names:
        names = ['NI' for i in range(len(points))]
    if not resnums:
        resnums = [0 for i in range(len(points))]
    chains = ['A' for i in range(len(points))]
    mm = pr.AtomGroup(title)
    mm.setCoords(points)
    mm.setResnums(resnums)
    mm.setNames(names)
    mm.setResnames(names)
    mm.setChids(chains)
    if resnames:
        mm.setResnames(resnames)
    return mm


def write2pymol(points, outdir, filename, names = None):
    '''
    Speically used for Metal binding geometry pdb output.
    '''
    mm = transfer2pdb(points, names)
    pr.writePDB(outdir + filename + '.pdb', mm)


def ordered_sel(mobile, target, mob_sel, target_sel):
    '''
    The prody sel always follow the order of atom index. 
    Here is to follow the selection order for alignment.
    mobile: prody object.
    target, prody object.
    mob_sel: list of atom names.
    target_sel: list of atom names.
    '''
    mobile_sel_coords = []
    for s in mob_sel.split(' '):
        mobile_sel_coords.append(mobile.select('name ' + s).getCoords()[0])
    
    target_sel_coords = []
    for s in target_sel.split(' '):
        target_sel_coords.append(target.select('name ' + s).getCoords()[0])

    return np.array(mobile_sel_coords), np.array(target_sel_coords)


def ordered_sel_transformation(mobile, target, mob_sel, target_sel):
    mobile_sel_coords, target_sel_coords = ordered_sel(mobile, target, mob_sel, target_sel)
    transformation = pr.calcTransformation(mobile_sel_coords, target_sel_coords)
    transformation.apply(mobile)
    rmsd = pr.calcRMSD(mobile_sel_coords, target_sel_coords)
    return rmsd


def ordered_sel_rmsd(mobile, target, mob_sel, target_sel):
    mobile_sel_coords, target_sel_coords = ordered_sel(mobile, target, mob_sel, target_sel)
    rmsd = pr.calcRMSD(mobile_sel_coords, target_sel_coords)
    return rmsd


def combine_vdm_into_ag(vdms, tag, geometry, overlapScore = 0, cluScore = 0, ideal_geometry = None):
    '''
    Merge all vdms into one prody AtomGroup.
    Generally for CombInfo.centroid_dict.
    '''
    ag = pr.AtomGroup(tag)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []
    betas = []
    occu = []
    chain_num = 0
    for v in vdms:
        c = v.query.select('not name NI MN ZN CO CU MG FE' )
        c.setChids(string.ascii_uppercase[chain_num])
        coords.extend(c.getCoords())
        chids.extend(c.getChids())
        names.extend(c.getNames())
        resnames.extend(c.getResnames())
        resnums.extend(c.getResnums())
        betas.extend([overlapScore for x in range(len(c))])
        occu.extend([cluScore for x in range(len(c))])
        chain_num += 1

    if not geometry:
        metal_center = pr.calcCenter([v.select('name NI MN ZN CO CU MG FE')[0].getCoords() for v in vdms])
        geometry = transfer2pdb(metal_center, ['NI'])
    geometry.setChids(string.ascii_uppercase[chain_num])
    _geo = geometry.select('name NI')
    coords.extend(_geo.getCoords())
    chids.extend(_geo.getChids())
    names.extend(_geo.getNames())
    resnames.extend(_geo.getResnames())
    resnums.extend(_geo.getResnums())
    betas.append(overlapScore)
    occu.append(cluScore)
    chain_num += 1

    if ideal_geometry:     
        ideal_geometry.setChids(string.ascii_uppercase[chain_num])
        coords.extend(ideal_geometry.getCoords())
        chids.extend(ideal_geometry.getChids())
        names.extend(ideal_geometry.getNames())
        resnames.extend(ideal_geometry.getResnames())
        resnums.extend(ideal_geometry.getResnums())
        betas.extend([overlapScore for i in range(len(ideal_geometry))])
        occu.extend([cluScore for i in range(len(ideal_geometry))])

    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    ag.setBetas(betas)
    ag.setOccupancies(occu)
    return ag


def combine_vdm_target_into_ag(target, resind_vdm_dict, write_geo, geometry, title, aa = 'ALA'):
    '''
    combine vdms into target and mutate all other aa 2 ala or gly for ligand position.
    '''

    ag = pr.AtomGroup(title)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []

    if aa == 'GLY':
        bb_sel = 'name N CA C O H'
    elif aa == 'ALA':
        bb_sel = 'name N CA C O H CB'
    else:
        bb_sel = ''

    for i in target.select('protein and name C').getResindices():
        c = target.select('resindex ' + str(i)  + ' ' + bb_sel)

        if i in resind_vdm_dict.keys():
            v = resind_vdm_dict[i]
            query = v.query.select('resindex ' + str(v.contact_resind))
            query.setChids([c.getChids()[0] for x in range(len(query))])
            query.setResnums([c.getResnums()[0] for x in range(len(query))])
            c = query

        else:
            if aa == 'GLY':
                c.setResnames(['GLY' for u in range(len(c))])
            elif aa == 'ALA':
                '''
                Add CB for gly in original structure.
                '''
                ala = constant.ideal_ala.copy()
                pr.calcTransformation(ala.select('name N CA C'), c.select('name N CA C')).apply(ala)
                ala.setChids([c.getChids()[0] for x in range(len(ala))])
                ala.setResnums([c.getResnums()[0] for x in range(len(ala))])
                c = ala.select(bb_sel)
            
        coords.extend(c.getCoords())
        chids.extend(c.getChids())
        names.extend(c.getNames())
        resnames.extend(c.getResnames())
        resnums.extend(c.getResnums())

    if write_geo:
        if not geometry:
            metal_center = pr.calcCenter([v.select('name NI MN ZN CO CU MG FE')[0].getCoords() for v in resind_vdm_dict.values()])
            geometry = transfer2pdb(metal_center, ['NI'])
        geometry.setChids(string.ascii_uppercase[len(np.unique(target.select('protein').getChids()))])
        _geo = geometry.select('name NI')
        coords.extend(_geo.getCoords())
        chids.extend(_geo.getChids())
        names.extend(_geo.getNames())
        resnames.extend(_geo.getResnames())
        resnums.extend(_geo.getResnums())

    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    return ag


def target_to_all_gly_ala(target, title, win_no_mutation = [], aa = 'ALA', keep_no_protein = False):
    '''
    For the prody object target, mutate all the aa into ala or gly. 
    '''
    ag = pr.AtomGroup(title)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []
    betas = []
    occu = []

    if aa == 'GLY':
        bb_sel = 'name N CA C O H'
    if aa == 'ALA':
        bb_sel = 'name N CA C O H CB'
    
    for i in target.select('protein and name C').getResindices():
        if i in win_no_mutation:
            c = target.select('resindex ' + str(i))
        else:
            c = target.select('resindex ' + str(i)  + ' ' + bb_sel)
            if aa == 'GLY':
                c.setResnames(['GLY' for i in range(len(c))])
            elif aa == 'ALA':
                '''
                Add CB for gly in original structure.
                '''
                ala_heavy = constant.ideal_ala.select('heavy').copy()
                pr.calcTransformation(ala_heavy.select('name N CA C'), c.select('name N CA C')).apply(ala_heavy)
                ala_heavy.setChids([c.getChids()[0] for x in range(len(ala_heavy))])
                ala_heavy.setResnums([c.getResnums()[0] for x in range(len(ala_heavy))])
                for x in ['N', 'CA', 'C', 'O', 'CB']:
                    if c.select('name ' + x) is not None:
                        ala_heavy.select('name ' + x).setCoords(c.select('name ' + x).getCoords())
                c = ala_heavy.select(bb_sel)

        coords.extend(c.getCoords())
        chids.extend(c.getChids())
        names.extend(c.getNames())
        resnames.extend(c.getResnames())
        resnums.extend(c.getResnums())
        betas.extend([0 for x in range(len(c))])
        occu.extend([0 for x in range(len(c))])
    if keep_no_protein:
        x = target.select('not protein')
        if x:           
            coords.extend(x.getCoords())
            chids.extend(x.getChids())
            names.extend(x.getNames())
            resnames.extend(x.getResnames())
            resnums.extend(x.getResnums())
            betas.extend([0 for x in range(len(x))])
            occu.extend([0 for x in range(len(x))])
    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    ag.setBetas(betas)
    ag.setOccupancies(occu)
    return ag

def target_mutation(target, title, win_to_mutation = [], aa = 'ALA', keep_no_protein = False):
    '''
    For the prody object target, mutate the selected aa into ala or gly. 
    Currently only work for ala or gly as their sidechains are more defined.
    '''
    ag = pr.AtomGroup(title)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []
    betas = []
    occu = []

    if aa == 'GLY':
        bb_sel = 'name N CA C O H'
    if aa == 'ALA':
        bb_sel = 'name N CA C O H CB'
    
    for i in target.select('protein and name C').getResindices():
        if i not in win_to_mutation:
            c = target.select('resindex ' + str(i))
        else:
            try:
                if aa == 'GLY':
                    c = target.select('resindex ' + str(i)  + ' ' + bb_sel)
                    c.setResnames(['GLY' for i in range(len(c))])
                elif aa == 'ALA':
                    '''
                    Add CB for gly in original structure.
                    '''
                    ala = constant.ideal_ala.copy()
                    pr.calcTransformation(ala.select('name N CA C'), c.select('name N CA C')).apply(ala)
                    ala.setChids([c.getChids()[0] for x in range(len(ala))])
                    ala.setResnums([c.getResnums()[0] for x in range(len(ala))])
                    c = ala.select(bb_sel)
            except:
                print('Failed mutation at resindex {}'.format(i))
                c = target.select('resindex ' + str(i))
        coords.extend(c.getCoords())
        chids.extend(c.getChids())
        names.extend(c.getNames())
        resnames.extend(c.getResnames())
        resnums.extend(c.getResnums())
        betas.extend([0 for x in range(len(c))])
        occu.extend([0 for x in range(len(c))])
    if keep_no_protein:
        x = target.select('not protein')
        coords.extend(x.getCoords())
        chids.extend(x.getChids())
        names.extend(x.getNames())
        resnames.extend(x.getResnames())
        resnums.extend(x.getResnums())
        betas.extend([0 for x in range(len(x))])
        occu.extend([0 for x in range(len(x))])
    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    ag.setBetas(betas)
    ag.setOccupancies(occu)
    return ag

def mutate_vdm_target_into_ag(target, resind_vdm_dict, title, write_metal = True, contact_resind = 1):
    '''
    combine vdms into target and mutate all other aa 2 ala or gly for ligand position.
    '''

    ag = pr.AtomGroup(title)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []

    for i in target.select('protein and name C').getResindices():
        c = target.select('resindex ' + str(i))

        if i in resind_vdm_dict.keys():
            v_query = resind_vdm_dict[i]
            query = v_query.select('resindex ' + str(contact_resind))
            query.setChids([c.getChids()[0] for x in range(len(query))])
            query.setResnums([c.getResnums()[0] for x in range(len(query))])
            c = query
            
        coords.extend(c.getCoords())
        chids.extend(c.getChids())
        names.extend(c.getNames())
        resnames.extend(c.getResnames())
        resnums.extend(c.getResnums())

    if write_metal:
        _geo = list(resind_vdm_dict.values())[0].select('name NI MN ZN CO CU MG FE')
        coords.extend(_geo.getCoords())
        chids.extend(_geo.getChids())
        names.extend(_geo.getNames())
        resnames.extend(_geo.getResnames())
        resnums.extend(_geo.getResnums())

    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    return ag


def mutate_vdm_target_into_ag2(target, ind_vdm_dict, title, vdm_sel):
    '''
    mutate vdms into target.
    target: protein bb, such as something like a std helix.
    ind_vdm_dict: resind in target where we want to put the vdms on, and the corresponding vdm.
    title: the atomgroup name.
    vdm_sel: 'chid X and resnum 10'
    '''

    ag = pr.AtomGroup(title)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []

    for i in np.unique(target.getResindices()):
        c = target.select('resindex ' + str(i))

        if i in ind_vdm_dict.keys():
            v_query = ind_vdm_dict[i]
            query = v_query.select(vdm_sel)
            # print(i)
            # print(v_query.getChids())
            # print(query)
            query.setChids([c.getChids()[0] for x in range(len(query))])
            query.setResnums([c.getResnums()[0] for x in range(len(query))])
            c = query
            
        coords.extend(c.getCoords())
        chids.extend(c.getChids())
        names.extend(c.getNames())
        resnames.extend(c.getResnames())
        resnums.extend(c.getResnums())

    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    return ag


def combine_ags(ags, title, ABCchids = None):
    
    ag = pr.AtomGroup(title)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []
    if not ABCchids:
        ABCchids = [chr(i) for i in range(65, 66 + len(ags))]
    chid_ind = 0
    for _ag_all in ags:
        for cd in np.unique(_ag_all.getChids()):
            #print(cd)
            chid = ABCchids[chid_ind]
            chid_ind += 1
            if cd == None:
                _ag = _ag_all
            else:
                _ag = _ag_all.select('chid ' + cd)
                
            for i in np.unique(_ag.getResindices()):
                c = _ag.select('resindex ' + str(i))
                coords.extend(c.getCoords())
                chids.extend([chid for x in range(len(c))])
                names.extend(c.getNames())
                resnames.extend(c.getResnames())
                resnums.extend(c.getResnums())

    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    return ag

def combine_ags_into_one_chain(ags, title):
    
    ag = pr.AtomGroup(title)
    coords = []
    chids = []
    names = []
    resnames = []
    resnums = []
    resnum = 1
    for _ag_all in ags:
        for cd in np.unique(_ag_all.getChids()):
            #print(cd)
            chid = 'A'

            if cd == None:
                _ag = _ag_all
            else:
                _ag = _ag_all.select('chid ' + cd)
                
            for i in np.unique(_ag.getResindices()):
                c = _ag.select('resindex ' + str(i))
                coords.extend(c.getCoords())
                chids.extend([chid for x in range(len(c))])
                names.extend(c.getNames())
                resnames.extend(c.getResnames())
                resnums.extend([resnum for x in range(len(c))])
                resnum += 1

    ag.setCoords(np.array(coords))
    ag.setChids(chids)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    return ag