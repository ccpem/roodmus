# helper functions to perform simulation analysis

# imports
import os
import re
import gemmi
import pytraj
import numpy as np
from emmer.ndimage.compute_real_space_correlation import compute_real_space_correlation
from emmer.pdb.convert.convert_pdb_to_map import convert_pdb_to_map
from emmer.ndimage.filter.low_pass_filter import low_pass_filter
from simtk import unit
from openmm.app import PDBFile, Modeller
from tqdm import tqdm

## 0. loading and saving
# retrieve simulation stats from logfile
# filename = os.path.join(outputdir, list_of_simulations[sim_idx], logfile)
class simstats(object):
    def __init__(self):
        self.w_exp = 0
        self.temperature = 0
        self.totaltime = 0
        self.ramptime = 0
        self.ramptimeT = 0
        self.calibrationtime = 0
        self.dt_update = 0
        self.targetmap = 0
        self.starting_model = 0
        self.raw_model = 0
        self.starting_rscc = 0

    def get_sim_stats(self, filename):
        fid = open(filename)
        lns = fid.readlines()
        fid.close()

        for ln in lns:
            if "running simulation on GPU" in ln:
                self.GPUID = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", ln)
            if "experimental restraint strength (w):" in ln:
                self.w_exp = float(ln.strip().split(":")[1])
            if "restraint strength ramp time" in ln:
                self.ramptime = float(ln.strip().split(":")[1])
            if "temperature ramp time [ps]" in ln:
                self.ramptimeT = float(ln.strip().split(":")[1])
            if "temperature:" in ln:
                self.temperature = float(ln.strip().split(":")[1])
            if "total simulation time [ps]:" in ln:
                self.totaltime = float(ln.strip().split(":")[1])
            if "calibration time [ps]:" in ln:
                self.calibrationtime = float(ln.strip().split(":")[1])
            if "time between experimental restraint update [ps]:" in ln:
                self.dt_update = float(ln.strip().split(":")[1])
            if "dt [ps]" in ln:
                self.dt = float(ln.strip().split(":")[1])
            if "target map file:" in ln:
                self.targetmap = ln.strip().split(":")[1].strip()
            if "starting model file:" in ln:
                self.starting_model = ln.strip().split(":")[1].strip()
                if os.path.isfile(self.starting_model.replace("_MD","")):
                    self.raw_model = self.starting_model.replace("_MD","")
                else:
                    self.raw_model = self.starting_model
            if "target map resolution" in ln:
                self.resolution = float(ln.strip().split(":")[1])
            if "map-to-model cross-correlation:" in ln:
                self.starting_rscc = float(ln.strip().split(":")[1])

def convert_traj_to_pdb(traj, starting_model_gemmi, numModels, outfilename):
    output_struct = gemmi.Structure()
    mdlidx = 1
    for frame in tqdm(range(0, len(traj), max(int(len(traj)/numModels),1))):
        atmidx = 0
        A = traj[frame].xyz
        output_struct.add_model(starting_model_gemmi[0])
        output_struct[-1].name = f"{mdlidx}"
        mdlidx += 1
        for chn in output_struct[-1]:
            for res in chn:
                for atm in res:
                    atm.pos.x = A[atmidx,0]
                    atm.pos.y = A[atmidx,1]
                    atm.pos.z = A[atmidx,2]
                    atmidx+=1

    output_struct.write_pdb(outfilename)
    ensemble_model_gemmi = gemmi.read_structure(outfilename)
    return ensemble_model_gemmi
                
def get_Bfactor(starting_model_gemmi, starting_model_openmm):
    numResidues = 0
    for mdl in starting_model_gemmi:
        for chn in mdl:
            for res in chn:
                numResidues += 1

    atoms = list(starting_model_openmm.topology.atoms())
    residues = list(starting_model_openmm.topology.residues())
    Bfac_atom = np.recarray(shape=(starting_model_gemmi[0].count_atom_sites()), dtype=[('name', object), ('residue', object), ('resid', int), ('b_iso', float), ('atmid', float)])
    Bfac_residue = np.recarray(shape=(numResidues), dtype=[('name', object), ('b_iso', float), ('resid', float)])
    atmidx = 0
    residx = 0
    for mdl in starting_model_gemmi:
        for chn in mdl:
            for res in chn:
                Btmp = []
                for atm in res:
                    Bfac_atom[atmidx].name = atm.name
                    Bfac_atom[atmidx].residue = res.name
                    Bfac_atom[atmidx].resid = residx
                    Bfac_atom[atmidx].b_iso = atm.b_iso
                    Bfac_atom[atmidx].atmid = atoms[atmidx].index
                    Btmp.append(atm.b_iso)
                    atmidx += 1
                Bfac_residue[residx].name = res.name
                Bfac_residue[residx].b_iso = np.mean(Btmp)
                Bfac_residue[residx].resid = residues[residx].index
                residx += 1
    return Bfac_atom, Bfac_residue
        
def revert_model_to_raw(model_openmm, raw_model_openmm, return_trimmed_model=True):
    atoms_not_in_raw = []
    for atm_model in model_openmm.topology.atoms():
        atom_in_raw_structure = False
        for atm_raw in raw_model_openmm.topology.atoms():
            if atm_model.name == atm_raw.name and \
            atm_model.element == atm_raw.element and \
            atm_model.residue.name == atm_raw.residue.name and \
            atm_model.residue.chain.id == atm_raw.residue.chain.id:
                atom_in_raw_structure = True
                break
        if not atom_in_raw_structure:
            atoms_not_in_raw.append(atm_model)

    atom_ids_not_in_raw = [r.index for r in atoms_not_in_raw]
    if return_trimmed_model:
        m = Modeller(model_openmm.topology, model_openmm.positions)
        m.delete(atoms_not_in_raw)
        return atom_ids_not_in_raw, m
    else:
        return atom_ids_not_in_raw
    
def convert_openmm_to_gemmi(model_openmm):
    model_gemmi = gemmi.Model("0")
    atmidx = 0
    for chain in model_openmm.topology.chains():
        chain_gemmi = gemmi.Chain(chain.id)
        for residue in chain.residues():
            residue_gemmi = gemmi.Residue()
            residue_gemmi.name = residue.name
            residue_gemmi.seqid = gemmi.SeqId(residue.id)
            for atom in residue.atoms():
                atom_gemmi = gemmi.Atom()
                atom_gemmi.b_iso = 0
                atom_gemmi.element = gemmi.Element(atom.element.symbol)
                atom_gemmi.name = atom.name
                atom_gemmi.occ = 1
                atom_gemmi.pos = gemmi.Position(model_openmm.positions[atmidx].value_in_unit(unit.angstrom).x, model_openmm.positions[atmidx].value_in_unit(unit.angstrom).y, model_openmm.positions[atmidx].value_in_unit(unit.angstrom).z)
                atmidx += 1
                residue_gemmi.add_atom(atom_gemmi)
            chain_gemmi.add_residue(residue_gemmi)
        model_gemmi.add_chain(chain_gemmi)
    structure_gemmi = gemmi.Structure()
    structure_gemmi.add_model(model_gemmi)
    ucell_openmm = model_openmm.topology.getUnitCellDimensions().value_in_unit(unit.angstrom)
    structure_gemmi.cell = gemmi.UnitCell(a=ucell_openmm.x, b=ucell_openmm.y, c=ucell_openmm.z, alpha=90, beta=90, gamma=90)
    return structure_gemmi

## 1.1 map-to-model real space cross-correlation as function of time
def get_starting_rscc(raw_model_gemmi, targetmap_gemmi, resolution):
    size = targetmap_gemmi.grid.shape
    voxelsize = targetmap_gemmi.grid.spacing[0]
    for mdl in raw_model_gemmi:
        for chn in mdl:
            for res in chn:
                for atm in res:
                    atm.b_iso = 0
    map_from_model_unfiltered, grid = convert_pdb_to_map(raw_model_gemmi, size=size, apix=voxelsize, return_grid=True)
    map_from_model = low_pass_filter(map_from_model_unfiltered, resolution, voxelsize)
    map_from_model = np.rot90(np.flip(map_from_model, axis=2), axes=(0,2))
    rscc = compute_real_space_correlation(map_from_model, np.array(targetmap_gemmi.grid))
    return rscc

# open log rscc logfile
def get_rscc_from_file(filename):
    fid = open(filename)
    header = fid.readline()
    lns = fid.readlines()
    fid.close()

    print(header)

    RSCC = []
    RSCC_ens = []
    time = []
    for ln in lns:
        a = ln.replace("\n","").split(',')
        RSCC.append(float(a[header.strip().split(",").index('"rscc"')]))
        RSCC_ens.append(float(a[header.strip().split(",").index('"ensemble rscc"')]))
        time.append(float(a[header.strip().split(',').index('"Time (ps)"')]))
    return RSCC, RSCC_ens, time

def compute_rscc_from_trajectory(traj, targetmap_gemmi, atom_ids_not_in_raw, resolution, time, calibrationtime, ramptime, ramptimeT, outputfilename, stepsize=1):
    size = np.array(targetmap_gemmi.grid).shape
    vsize = targetmap_gemmi.grid.spacing[0]
    unitcell = targetmap_gemmi.grid.unit_cell
    rscc_log = np.recarray(shape=(int(traj.n_frames/stepsize)+1), dtype=[("phase", object), ("time", float), ("rscc", float), ("rscc_ens", float)])
    ensemble_map = np.zeros(targetmap_gemmi.grid.shape)
    idx = 0
    start_ensemble = False
    for fidx in tqdm(range(0, traj.n_frames, stepsize)):
        pytraj.io.write_traj(outputfilename, traj, format="PDB", frame_indices=[fidx], overwrite=True)

        single_frame_openmm = PDBFile(outputfilename)
        m = Modeller(single_frame_openmm.topology, single_frame_openmm.positions)
        atoms_not_in_raw = [r for r in m.topology.atoms() if r.index in atom_ids_not_in_raw]
        m.delete(atoms_not_in_raw)
        single_frame_gemmi = convert_openmm_to_gemmi(m)
        single_frame_gemmi.remove_hydrogens()

        map_from_model_unfiltered = convert_pdb_to_map(input_pdb = single_frame_gemmi, unitcell = unitcell, size = size, return_grid=False)
        map_from_model_zyx = low_pass_filter(map_from_model_unfiltered, resolution, vsize)
        map_from_model = np.rot90(np.flip(map_from_model_zyx,axis=0),axes=(2,0))

        if start_ensemble:
            ensemble_map += map_from_model

        rscc = compute_real_space_correlation(map_from_model, np.array(targetmap_gemmi.grid))
        rscc_ens = compute_real_space_correlation(ensemble_map, np.array(targetmap_gemmi.grid))

        if time[fidx] < calibrationtime:
            phase = "calibration"
        elif time[fidx] < (ramptime + calibrationtime):
            phase = "ramp"
        elif time[fidx] < (ramptimeT + ramptime + calibrationtime):
            phase = "temperature"
            start_ensemble = True
        else:
            phase = "running"    

        rscc_log[idx].phase = phase
        rscc_log[idx].time = time[fidx]
        rscc_log[idx].rscc = rscc
        rscc_log[idx].rscc_ens = rscc_ens
        idx += 1
    return rscc_log
        
## 1.2 map-to-model real-space cross-correlation per residue
def compute_rscc_per_residue(targetmap_gemmi, starting_model_gemmi, ensemble_model_gemmi, resolution, stepsize=1):
    numResidues = 0
    for mdl in starting_model_gemmi:
        for chn in mdl:
            for res in chn:
                numResidues += 1

    size = targetmap_gemmi.grid.shape
    voxelsize = targetmap_gemmi.grid.spacing[0]

    residues = np.recarray(shape=(numResidues*len(ensemble_model_gemmi)), dtype=[("frame", int), ("resid", int), ("residue", object)])
    idx = 0
    for mdlidx, mdl in enumerate(ensemble_model_gemmi):
        residues[idx].frame = mdlidx
        for chn in mdl:
            for res in chn:
                residues[idx].residue = res
                residues[idx].resid = idx
                idx += 1

    rscc_per_residue = []
    for resid in tqdm(range(0, numResidues, stepsize)):
        # create new gemmi structure to hold single residue only
        single_residue_structure_gemmi = gemmi.Structure()
        single_residue_model_gemmi = gemmi.Model("1")
        single_residue_chain_gemmi = gemmi.Chain("A")

        for res in residues[residues.resid == resid]:
            single_residue_chain_gemmi.add_residue(res.residue)
        single_residue_model_gemmi.add_chain(single_residue_chain_gemmi)
        single_residue_structure_gemmi.add_model(single_residue_model_gemmi)

        map_from_model_unfiltered, grid = convert_pdb_to_map(single_residue_structure_gemmi, size=size, apix=voxelsize, return_grid=True)
        map_from_model = low_pass_filter(map_from_model_unfiltered, resolution, voxelsize)
        map_from_model = np.rot90(np.flip(map_from_model, axis=2), axes=(0,2))
        rscc = compute_real_space_correlation(map_from_model, np.array(targetmap_gemmi.grid))
        rscc_per_residue.append([resid, rscc])
    return residues, rscc_per_residue
    
    
def save_array_as_chimera_attribute(pdbfilename, values, attribute_name, recipient, filename, mode="Chimera"):
    fid = open(filename,'w')
    fid.write(f"attribute: {attribute_name}\n")
    fid.write(f"recipient: {recipient}\n")
    fid.write("match mode: 1-to-1\n")
    
    pdb_gemmi = gemmi.read_structure(pdbfilename)
    pdb_text = open(pdbfilename)
    lns = pdb_text.readlines()
    pdb_text.close()
#     print(lns[100][:4])
#     print(lns[100][17:20])
#     print(lns[100][21])
        
    if len(values) == pdb_gemmi[0].count_atom_sites() and recipient == "residues":
        new_values = []
        atmidx = 0
        for chn in pdb_gemmi[0]:
            for res in chn:
                tmp = []
                for atm in res:
                    tmp.append(values[atmidx])
                    atmidx += 1
                new_values.append(np.mean(tmp, axis=0))
        values = new_values
    
    if recipient == "atoms":
        atmidx = 0
        for mdl in pdb_gemmi:
            for chn in mdl:
                for res in chn:
                    for atm in res:
                        if mode == "Chimera":
                            atom_id = f":.{chn.name}:{res.name}@{atm.name}\t"
                        elif mode == "ChimeraX":
                            atom_id = f"/{chn.name}:{res.name}@{atm.name}\t"
#                         print(atom_id+str(values[atmidx][1]))
                        fid.write("\t"+atom_id+str(values[atmidx])+"\n")
                        atmidx += 1

    elif recipient == "residues":
        residx = lnidx = idx = 0
        for mdl in pdb_gemmi:
            for chn in mdl:
                for res in chn:
                    tmp = [r for r in lns[lnidx:] if r[:4]=="ATOM" and r[17:20]==res.name and r[21]==chn.name]
#                     print(tmp[-1])
#                     print(res.name)
#                     print(chn.name)
                    residx = int(tmp[0][22:27])
                    tmp = [r for r in lns[lnidx:] if r[:4]=="ATOM" and r[17:20]==res.name and r[21]==chn.name and int(r[22:27])==residx]
                    lnidx = lns.index(tmp[-1])
#                     print(lnidx)
                        
                    if mode == "Chimera":
                        res_id = f":.{chn.name}:{residx}\t"
                    elif mode == "ChimeraX":
                        res_id = f"/{chn.name}:{residx}\t"
#                     print(atom_id+str(values[atmidx][1]))
                    fid.write("\t"+res_id+str(values[idx])+"\n")
                    idx += 1

    fid.close()

