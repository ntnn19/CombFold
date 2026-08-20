import dataclasses
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.spatial.distance

import Bio.SeqUtils
import Bio.PDB, Bio.PDB.Residue

from libs.prepare_complex import create_complexes
from libs.utils_classes import read_subunits_info, SubunitName, SubunitsInfo, INTERFACE_MIN_ATOM_DIST
from libs.utils_pdb import get_pdb_model_readonly, copy_pdb_set_start_offset, copy_pdb_rename_chain


THIS_SCRIPT_PATH = os.path.abspath(__file__)
BASE_PATH = os.path.dirname(THIS_SCRIPT_PATH)

BINARY_PATH = os.path.abspath(os.path.join(BASE_PATH, "..", "CombinatorialAssembler"))
AF2TRANS_BIN_PATH = os.path.join(BINARY_PATH, "AF2trans.out")
COMB_ASSEMBLY_BIN_PATH = os.path.join(BINARY_PATH, "CombinatorialAssembler.out")


# In most cases PartialSubunit will be the complete subunit, but this allows to input PDBs of interactions with only the
# interfaces between subunits.
@dataclasses.dataclass
class PartialSubunit:
    subunit_name: str
    pdb_path: str
    chain_id: str
    start_residue_id: int  # inclusive
    end_residue_id: int  # inclusive
    subunit_start_sequence_id: int
    is_complete: bool = False
    plddt: Optional[float] = None  # mean CA bfactor over the subunit range, computed at scan time


@dataclasses.dataclass
class TransformationInfo:
    subunit_names: Tuple[str, str]
    pdb_path: str
    pdb_chain_ids: Tuple[str, str]
    rep_imposed_rmsds: Tuple[float, float]
    transformation_numbers: str
    score: float


def get_chain_to_seq(pdb_path: str, use_seqres: bool = True) -> Dict[str, str]:
    if use_seqres:
        chain_to_seq = {str(record.id): str(record.seq) for record in Bio.SeqIO.parse(pdb_path, 'pdb-seqres')}
        if len(chain_to_seq) > 0:
            return chain_to_seq

    model = get_pdb_model_readonly(pdb_path)
    chain_to_seq = {}
    for chain in model.get_chains():
        res_id_to_res = {res.get_id()[1]: res for res in chain.get_residues() if "CA" in res}
        if len(res_id_to_res) == 0:
            print("skipping empty chain", chain.get_id())
            continue
        chain_to_seq[chain.get_id()] = ""
        for i in range(1, max(res_id_to_res) + 1):
            if i in res_id_to_res:
                chain_to_seq[chain.get_id()] += Bio.SeqUtils.seq1(res_id_to_res[i].get_resname())
            else:
                chain_to_seq[chain.get_id()] += "X"
    return chain_to_seq


def _iter_pdb_ca_records(pdb_path: str):
    """Fast single-pass scan of a PDB file yielding one record per CA atom:
    (chain_id, res_seq, res_name, (x, y, z), bfactor).

    Mirrors what Bio.PDB.PDBParser exposes for the first model (ATOM + HETATM
    records, fixed columns), but avoids building the full Structure object --
    parsing with Bio.PDB is the bottleneck when the PDB folder has many files.
    """
    n_models = 0
    seen_endmdl = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                n_models += 1
                if n_models > 1:
                    raise AssertionError(f"Too many models! {pdb_path}")
            elif line.startswith("ENDMDL"):
                seen_endmdl = True  # only the first model is used, like get_pdb_model_readonly
            elif (line.startswith("ATOM") or line.startswith("HETATM")) and not seen_endmdl:
                if line[12:16].strip() != "CA":
                    continue
                if line[16] not in (" ", "A"):  # skip alternate locations, like Bio.PDB
                    continue
                try:
                    bfactor = float(line[60:66])
                except ValueError:
                    bfactor = 0.0
                yield (line[21], int(line[22:26]), line[17:20].strip(),
                       (float(line[30:38]), float(line[38:46]), float(line[46:54])), bfactor)


def _scan_pdb_ca_atoms(pdb_path: str) -> Dict[str, Dict[int, Tuple[str, float]]]:
    """One fast pass over a PDB file: chain_id -> {res_seq: (res_name, ca_bfactor)}.

    Equivalent to parsing with Bio.PDB and keeping residues that have a CA atom
    (later records overwrite earlier ones with the same residue number, matching
    the dict comprehension in get_chain_to_seq), but much faster.
    """
    chains: Dict[str, Dict[int, Tuple[str, float]]] = defaultdict(dict)
    for chain_id, res_seq, res_name, _coord, bfactor in _iter_pdb_ca_records(pdb_path):
        chains[chain_id][res_seq] = (res_name, bfactor)
    return dict(chains)


def _chain_seq_from_ca_scan(res_id_to_ca: Dict[int, Tuple[str, float]]) -> str:
    """Replicates the sequence-building logic of get_chain_to_seq(use_seqres=False)."""
    return "".join(Bio.SeqUtils.seq1(res_id_to_ca[i][0]) if i in res_id_to_ca else "X"
                   for i in range(1, max(res_id_to_ca) + 1))


def _get_partial_subunit_residues(partial_subunit: PartialSubunit) -> List[Bio.PDB.Residue.Residue]:
    pdb_model = get_pdb_model_readonly(partial_subunit.pdb_path)
    return [res for res in pdb_model[partial_subunit.chain_id] if
            partial_subunit.start_residue_id <= res.id[1] <= partial_subunit.end_residue_id]


def extract_partial_subunit(partial_subunit: PartialSubunit, output_path: str):
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    pdb_struct = pdb_parser.get_structure("original_pdb", partial_subunit.pdb_path)
    assert len(list(pdb_struct)) == 1, "can't extract if more than one model"
    model = next(iter(pdb_struct))
    chains = list(model.get_chains())
    assert len([c for c in chains if c.id == partial_subunit.chain_id]) == 1, f"Missing: {partial_subunit.chain_id}"
    for chain in chains:
        if chain.id != partial_subunit.chain_id:
            model.detach_child(chain.id)

    # compute the residues to keep from the SAME parsed model. The original code re-parsed
    # the file a second time (via get_pdb_model_readonly) and matched residues across the two
    # parses by full_id equality in a list scan -- O(n_residues^2) full_id computations.
    res_ids_to_keep = {res.id for res in model[partial_subunit.chain_id]
                       if partial_subunit.start_residue_id <= res.id[1] <= partial_subunit.end_residue_id}
    res_to_remove = [res for res in model.get_residues() if res.id not in res_ids_to_keep]
    for res in res_to_remove:
        res.parent.detach_child(res.id)

    io = Bio.PDB.PDBIO()
    io.set_structure(pdb_struct)
    io.save(output_path)


def extract_partial_from_representative(partial_subunit: PartialSubunit, representative_subunits_path: str,
                                        output_folder: str, subunits_info: SubunitsInfo) -> str:
    subunit_info = subunits_info[partial_subunit.subunit_name]
    start_res_id = partial_subunit.subunit_start_sequence_id + subunit_info.start_res
    end_res_id = start_res_id + (partial_subunit.end_residue_id - partial_subunit.start_residue_id)

    output_pdb_path = os.path.join(output_folder, f"{partial_subunit.subunit_name}_{start_res_id}_"
                                                  f"{end_res_id}.pdb")
    if os.path.exists(output_pdb_path):
        return output_pdb_path

    chain_name, ident_subunit_name = subunit_info.chain_names[0], subunit_info.get_chained_names()[0]
    rep_subunit_path = os.path.join(representative_subunits_path, f"{ident_subunit_name}.pdb")

    rep_partial_subunit = PartialSubunit(subunit_name=subunit_info.name,
                                         pdb_path=rep_subunit_path,
                                         chain_id=chain_name,
                                         start_residue_id=start_res_id,
                                         end_residue_id=end_res_id,
                                         subunit_start_sequence_id=0)  # subunit_start_sequence_id is ignored
    extract_partial_subunit(rep_partial_subunit, output_pdb_path)
    return output_pdb_path


def score_transformation(pdb_path1: str, pdb_path2: str) -> Optional[float]:
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    model1 = next(iter(pdb_parser.get_structure("pdb1", pdb_path1)))
    model2 = next(iter(pdb_parser.get_structure("pdb2", pdb_path2)))
    chains1, chains2 = list(model1.get_chains()), list(model2.get_chains())
    assert len(chains1) == len(chains2) == 1, "can't extract if more than one chain"
    chain1, chain2 = chains1[0], chains2[0]

    chain1_res = [res for res in chain1.get_residues()]
    chain2_res = [res for res in chain2.get_residues()]

    chain1_ca = np.array([res["CA"].get_coord() for res in chain1_res])
    chain2_ca = np.array([res["CA"].get_coord() for res in chain2_res])

    close_residues = np.argwhere(scipy.spatial.distance.cdist(chain1_ca, chain2_ca) < INTERFACE_MIN_ATOM_DIST)
    if len(close_residues) == 0:
        print("Skipping transformation, missing interface between",
              os.path.basename(pdb_path1)[8:-4], os.path.basename(pdb_path2)[8:-4])
        return None

    chain1_interface, chain2_interface = set(), set()
    for i, j in close_residues:
        chain1_interface.add(i)
        chain2_interface.add(j)

    bfactors = [chain1_res[i]["CA"].get_bfactor() for i in chain1_interface] + \
               [chain2_res[i]["CA"].get_bfactor() for i in chain2_interface]
    return sum(bfactors) / len(bfactors)


@lru_cache(maxsize=4)
def _read_pdb_lines_and_ca(pdb_path: str):
    """Read a PDB file once and keep, per chain, its ATOM/HETATM lines and CA records.

    Returns chain_id -> (lines, ca_records) with ca_records = (res_seq, (x, y, z), bfactor).
    Caching the last few files avoids re-reading the same file for every pair of subunits
    found in it (the original code re-parsed the file with Bio.PDB twice per pair).
    """
    chains = defaultdict(lambda: ([], []))
    n_models = 0
    seen_endmdl = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                n_models += 1
                if n_models > 1:
                    raise AssertionError(f"Too many models! {pdb_path}")
            elif line.startswith("ENDMDL"):
                seen_endmdl = True
            elif (line.startswith("ATOM") or line.startswith("HETATM")) and not seen_endmdl:
                chain_id = line[21]
                chains[chain_id][0].append(line)
                if line[12:16].strip() == "CA" and line[16] in (" ", "A"):
                    try:
                        bfactor = float(line[60:66])
                    except ValueError:
                        bfactor = 0.0
                    chains[chain_id][1].append((int(line[22:26]),
                                                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                                                bfactor))
    return dict(chains)


def _write_sample_and_get_ca(partial_subunit: PartialSubunit, output_path: str):
    """Write the single-chain sample PDB for a partial subunit (same atom selection as
    extract_partial_subunit: one chain, residues within [start, end]) and return its
    CA records -- without ever building a Bio.PDB Structure."""
    chains = _read_pdb_lines_and_ca(partial_subunit.pdb_path)
    assert partial_subunit.chain_id in chains, f"Missing: {partial_subunit.chain_id}"
    lines, ca_records = chains[partial_subunit.chain_id]
    start, end = partial_subunit.start_residue_id, partial_subunit.end_residue_id
    with open(output_path, "w") as f:
        for line in lines:
            if start <= int(line[22:26]) <= end:
                f.write(line)
        f.write("END\n")
    return [ca for ca in ca_records if start <= ca[0] <= end]


def _score_ca_records(ca_records1, ca_records2) -> Optional[float]:
    """Interface pLDDT score, identical to score_transformation but computed in memory
    (the original re-parsed the two sample PDB files it had just written)."""
    if len(ca_records1) == 0 or len(ca_records2) == 0:
        return None
    chain1_ca = np.array([ca[1] for ca in ca_records1])
    chain2_ca = np.array([ca[1] for ca in ca_records2])

    close_residues = np.argwhere(scipy.spatial.distance.cdist(chain1_ca, chain2_ca) < INTERFACE_MIN_ATOM_DIST)
    if len(close_residues) == 0:
        return None

    chain1_interface, chain2_interface = set(), set()
    for i, j in close_residues:
        chain1_interface.add(i)
        chain2_interface.add(j)

    bfactors = [ca_records1[i][2] for i in chain1_interface] + \
               [ca_records2[j][2] for j in chain2_interface]
    return sum(bfactors) / len(bfactors)


def get_transformation_from_partials(partial_subunit1: PartialSubunit, partial_subunit2: PartialSubunit,
                                     representative_subunits_path: str, temp_folder: str,
                                     subunits_info: SubunitsInfo,
                                     rep_partials_folder: Optional[str] = None) -> Optional[TransformationInfo]:
    # rep partials depend only on the subunit and residue range, not on the sampled PDB --
    # extract them once per run into a persistent folder. (The original code extracted them
    # into temp_folder, which is wiped after every PDB file, causing a full re-extraction --
    # two Bio.PDB parses -- for every pair of every file.)
    rep_partials_folder = rep_partials_folder or temp_folder
    rep_struct1_path = extract_partial_from_representative(partial_subunit1, representative_subunits_path,
                                                           rep_partials_folder, subunits_info)
    rep_struct2_path = extract_partial_from_representative(partial_subunit2, representative_subunits_path,
                                                           rep_partials_folder, subunits_info)

    sample_struct1_path = os.path.join(temp_folder, f"sample1_{partial_subunit1.subunit_name}.pdb")
    ca_records1 = _write_sample_and_get_ca(partial_subunit1, sample_struct1_path)

    sample_struct2_path = os.path.join(temp_folder, f"sample2_{partial_subunit2.subunit_name}.pdb")
    ca_records2 = _write_sample_and_get_ca(partial_subunit2, sample_struct2_path)

    score = _score_ca_records(ca_records1, ca_records2)
    if score is None:
        print("Skipping transformation, missing interface between",
              partial_subunit1.subunit_name, partial_subunit2.subunit_name)
        return None

    af2trans_output = subprocess.check_output([AF2TRANS_BIN_PATH, rep_struct1_path, rep_struct2_path,
                                               sample_struct1_path, sample_struct2_path]).decode()
    assert af2trans_output.count(" | ") == 3, f"Unexpected output from AF2mer2trans {af2trans_output}"

    _, su1_desc, su2_desc, trans_nums = af2trans_output.split(" | ")
    rep_imposed_rmsds = (float(su1_desc.split("_")[0]), float(su2_desc.split("_")[0]))

    return TransformationInfo(
        subunit_names=(partial_subunit1.subunit_name, partial_subunit2.subunit_name),
        pdb_path=partial_subunit1.pdb_path,
        pdb_chain_ids=(partial_subunit1.chain_id, partial_subunit2.chain_id),
        transformation_numbers=trans_nums,
        rep_imposed_rmsds=rep_imposed_rmsds,
        score=score
    )


def get_pdb_to_partial_subunits(pdbs_folder: str, subunits_info: SubunitsInfo) -> Dict[str, List[PartialSubunit]]:
    pdb_path_to_partial_subunits: Dict[str, List[PartialSubunit]] = {}

    # for each pdb in folder
    for pdb_filename in os.listdir(pdbs_folder):
        if not pdb_filename.endswith(".pdb"):
            continue
        pdb_path = os.path.join(pdbs_folder, pdb_filename)

        partial_subunits: List[PartialSubunit] = []
        # one fast text pass per file instead of a full Bio.PDB parse
        ca_scan = _scan_pdb_ca_atoms(pdb_path)
        chain_to_seq = {chain_id: _chain_seq_from_ca_scan(res_id_to_ca)
                        for chain_id, res_id_to_ca in ca_scan.items()}
        for chain_id, chain_seq in chain_to_seq.items():
            for subunit_info in subunits_info.values():
                subunit_seq = subunit_info.sequence
                if subunit_seq in chain_seq:
                    print(f"found full {subunit_info.name} in {pdb_filename} chain {chain_id}")
                    start_res_id = chain_seq.index(subunit_seq) + 1
                    end_res_id = start_res_id + len(subunit_seq) - 1
                    # mean CA bfactor over the subunit range (used to pick the representative
                    # subunit later, saving a second parse of every file). Same summation order
                    # (file order) as extract_representative_subunits.
                    ca_bfactors = [res_name_bf[1] for res_id, res_name_bf in ca_scan[chain_id].items()
                                   if start_res_id <= res_id <= end_res_id]
                    plddt = sum(ca_bfactors) / len(ca_bfactors)
                    partial_subunits.append(PartialSubunit(subunit_name=subunit_info.name,
                                                           pdb_path=pdb_path,
                                                           chain_id=chain_id,
                                                           start_residue_id=start_res_id,
                                                           end_residue_id=end_res_id,
                                                           subunit_start_sequence_id=0,
                                                           is_complete=True,
                                                           plddt=plddt)
                                            )
                elif chain_seq in subunit_seq:
                    start_residue_id = 1
                    end_residue_id = len(chain_seq)
                    print(f"found partial {subunit_info.name} in {pdb_filename} chain {chain_id}"
                          f"{(end_residue_id - start_residue_id + 1)}/{len(subunit_seq)}")
                    partial_subunits.append(PartialSubunit(subunit_name=subunit_info.name,
                                                           pdb_path=pdb_path,
                                                           chain_id=chain_id,
                                                           start_residue_id=start_residue_id,
                                                           end_residue_id=end_residue_id,
                                                           subunit_start_sequence_id=subunit_seq.index(chain_seq),
                                                           is_complete=False)
                                            )
                else:
                    min_match_length = 10
                    start_ind = 0
                    while start_ind < len(chain_seq) - min_match_length:
                        if chain_seq[start_ind] == "X":
                            start_ind += 1
                            continue
                        end_ind = start_ind + min_match_length
                        while len(chain_seq[start_ind:end_ind].replace("X", "")) < min_match_length:
                            end_ind += 1
                            if end_ind >= len(chain_seq):
                                end_ind += 1
                                break
                        if end_ind >= len(chain_seq):
                            break

                        if chain_seq[start_ind:end_ind] not in subunit_seq:
                            start_ind += 1
                            continue
                        while end_ind < len(chain_seq) and chain_seq[start_ind:end_ind] in subunit_seq:
                            end_ind += 1
                        if chain_seq[start_ind:end_ind] not in subunit_seq:
                            end_ind -= 1

                        start_residue_id = start_ind + 1  # get_chain_seq function is 0-based, while res_id is 1-based
                        end_residue_id = end_ind + 1

                        subunit_start_ind = subunit_seq.index(chain_seq[start_ind:end_ind])
                        print(f"found small partial {subunit_info.name} in {pdb_filename} chain {chain_id} "
                              f"starting at index {start_ind} to {end_ind} (on subunit {subunit_start_ind})"
                              f"length {(end_ind - start_ind)}/{len(subunit_seq)}")
                        partial_subunits.append(PartialSubunit(subunit_name=subunit_info.name,
                                                               pdb_path=pdb_path,
                                                               chain_id=chain_id,
                                                               start_residue_id=start_residue_id,
                                                               end_residue_id=end_residue_id - 1,  # inclusive index
                                                               subunit_start_sequence_id=subunit_start_ind,
                                                               is_complete=False)
                                                )
                        start_ind = end_ind

        # print(f"found {len(partial_subunits)} partial subunits in {pdb_filename}")
        partial_subunits = sorted(partial_subunits, key=lambda x: (x.subunit_name, x.chain_id, x.start_residue_id))
        pdb_path_to_partial_subunits[pdb_path] = partial_subunits
    return pdb_path_to_partial_subunits


def extract_representative_subunits(pdb_path_to_partial_subunits: Dict[str, List[PartialSubunit]],
                                    subunits_info: SubunitsInfo, representative_subunits_path: str):
    rep_structs: Dict[SubunitName, Tuple[float, PartialSubunit]] = {}
    for pdb_path, partial_subunits in pdb_path_to_partial_subunits.items():
        for partial_subunit in partial_subunits:
            if not partial_subunit.is_complete:
                continue
            if partial_subunit.plddt is not None:
                # computed during the initial scan, no need to re-parse the file
                plddt_score = partial_subunit.plddt
            else:
                subunit_residues = _get_partial_subunit_residues(partial_subunit)
                plddt_score = sum([res["CA"].get_bfactor() for res in subunit_residues]) / len(subunit_residues)
            if rep_structs.get(partial_subunit.subunit_name, (-1, None))[0] < plddt_score:
                rep_structs[partial_subunit.subunit_name] = (plddt_score, partial_subunit)
    assert len(rep_structs) == len(subunits_info), "missing rep subunits for" + \
                                                   str(set(subunits_info.keys()) - set(rep_structs.keys()))
    for subunit_name, (plddt_score, partial_subunit) in rep_structs.items():
        print(f"rep {subunit_name} has plddt score {plddt_score}")
        subunit_info = subunits_info[subunit_name]
        rep_struct_path = os.path.join(representative_subunits_path, f"{subunit_name}.pdb")
        print("extracting partial subunit to", partial_subunit)
        extract_partial_subunit(partial_subunit, rep_struct_path)
        copy_pdb_set_start_offset(rep_struct_path, subunit_info.start_res, rep_struct_path)
        for chain_name, ident_subunit_name in zip(subunit_info.chain_names, subunit_info.get_chained_names()):
            copy_pdb_rename_chain(rep_struct_path, chain_name,
                                  os.path.join(representative_subunits_path, f"{ident_subunit_name}.pdb"))
        os.remove(rep_struct_path)


def extract_transformations(pdb_path_to_partial_subunits: Dict[str, List[PartialSubunit]], subunits_info: SubunitsInfo,
                            representative_subunits_path: str, transformations_path: str):
    temp_folder = os.path.join(transformations_path, "temp_transformations")
    # persistent cache for representative-subunit partials (survives the per-file temp cleanup)
    rep_partials_folder = os.path.join(transformations_path, "temp_rep_partials")
    os.makedirs(rep_partials_folder, exist_ok=True)
    transformations_by_pdb_path: Dict[str, List[TransformationInfo]] = {}
    for pdb_path, partial_subunits in pdb_path_to_partial_subunits.items():
        print("- Extracting pairwise transformations from file", pdb_path)
        if os.path.exists(temp_folder):
            print("removing temp folder")
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)
        transformations_by_pdb_path[pdb_path] = []
        for partial_subunit_ind_i in range(len(partial_subunits)):
            partial_subunit1 = partial_subunits[partial_subunit_ind_i]

            for partial_subunit_ind_j in range(partial_subunit_ind_i + 1, len(partial_subunits)):
                partial_subunit2 = partial_subunits[partial_subunit_ind_j]

                transformation_info = get_transformation_from_partials(partial_subunit1, partial_subunit2,
                                                                       representative_subunits_path, temp_folder,
                                                                       subunits_info, rep_partials_folder)
                if transformation_info is not None:
                    transformations_by_pdb_path[pdb_path].append(transformation_info)
        shutil.rmtree(temp_folder)
    shutil.rmtree(rep_partials_folder)

    transformations_by_subunit_pair: Dict[Tuple[str, str], List[TransformationInfo]] = defaultdict(list)
    for pdb_path, transformations in transformations_by_pdb_path.items():
        for transformation in transformations:
            transformations_by_subunit_pair[transformation.subunit_names].append(transformation)

    for (subunit_name1, subunit_name2), transformations in transformations_by_subunit_pair.items():
        print(f"found {len(transformations)} transformations between {subunit_name1} and {subunit_name2}")
        transformations = sorted(transformations, key=lambda x: x.score, reverse=True)
        output_file_path = os.path.join(transformations_path, f"tmp_{subunit_name1}_plus_{subunit_name2}")

        with open(output_file_path, "w") as f:
            for i, transformation in enumerate(transformations):
                description = f"{transformation.rep_imposed_rmsds[0]}_{transformation.rep_imposed_rmsds[1]}_" \
                              f"{transformation.pdb_chain_ids[0]}_{transformation.pdb_chain_ids[1]}_" \
                              f"{os.path.basename(transformation.pdb_path)}"
                f.write(f"{i + 1} | {transformation.score} | {description} | {transformation.transformation_numbers}\n")

        aliases_c1 = subunits_info[subunit_name1].get_chained_names()
        aliases_c2 = subunits_info[subunit_name2].get_chained_names()

        for c1 in range(len(aliases_c1)):
            start_from = c1 + 1 if subunit_name1 == subunit_name2 else 0
            for c2 in range(start_from, len(aliases_c2)):
                alias_output_path = os.path.join(transformations_path, f"{aliases_c1[c1]}_plus_{aliases_c2[c2]}")
                shutil.copy(output_file_path, alias_output_path)
        os.remove(output_file_path)


def run_combfold(representative_subunits_path: str, subunits_info: SubunitsInfo, transformations_path: str,
                 crosslinks_path: Optional[str], output_path: str, output_cif: bool = False,
                 max_results_number: int = 5, subunits_group1: Optional[List[str]] = None,):
    # prepare and run assembly
    with open(os.path.join(representative_subunits_path, "chain.list"), "w") as f:
        sorted_all_subunits = sorted(sum([i.get_chained_names() for i in subunits_info.values()], []))
        for chained_subunit_name in sorted_all_subunits:
            if subunits_group1 is not None and chained_subunit_name in subunits_group1:
                f.write(f"{chained_subunit_name}.pdb 1\n")
            else:
                f.write(f"{chained_subunit_name}.pdb\n")
    os.chdir(representative_subunits_path)
    if crosslinks_path is not None:
        shutil.copy(crosslinks_path, os.path.join(representative_subunits_path, "xlink_consts.txt"))
    else:
        open("xlink_consts.txt", "w").close()

    subprocess.run(f"{COMB_ASSEMBLY_BIN_PATH} chain.list {transformations_path}/ 900 100 xlink_consts.txt "
                   f"-b 0.05 -t 80 > output.log 2>&1", shell=True)
    print("--- Finished combinatorial assembly, writing output models")

    # build pdbs from assembly output
    clusters_path = os.path.join(representative_subunits_path, "output_clustered.res")
    if not os.path.exists(clusters_path):
        print(f"Could not assemble, exiting")
        return
    assembled_files = create_complexes(clusters_path, first_result=0, last_result=max_results_number,
                                       output_folder=os.path.join(output_path, "assembled_results"),
                                       output_cif=output_cif)

    confidence = []
    for result_as_str in open(clusters_path, "r").read().split("\n")[:len(assembled_files)]:
        if not result_as_str.strip():
            continue
        splitted_result = result_as_str.split(" ")
        confidence.append(float(splitted_result[splitted_result.index("weightedTransScore") + 1]))

    with open(os.path.join(output_path, "assembled_results", "confidence.txt"), "w") as f:
        for filename, c in zip(assembled_files, confidence):
            f.write(f"{filename} {c}\n")

    print(f"--- Assembled {len(assembled_files)} complexes, confidence: {min(confidence)}-{max(confidence)}")


def run_on_pdbs_folder(subunits_json_path: str, pdbs_folder: str, output_path: str,
                       crosslinks_path: Optional[str] = None, output_cif: bool = True, max_results_number: int = 5):
    pdbs_folder = os.path.abspath(pdbs_folder)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(COMB_ASSEMBLY_BIN_PATH):
        print(f"combinatorial assembly binary not found at {COMB_ASSEMBLY_BIN_PATH}, compile it by: \n"
              f"cd {os.path.dirname(COMB_ASSEMBLY_BIN_PATH)} && make")
        return

    if os.path.exists(output_path) and os.listdir(output_path):
        print(f"output path {output_path} is not empty, exiting")
        return

    subunits_info: SubunitsInfo = read_subunits_info(subunits_json_path)

    # representative_subunits_path is also the assembly algorithm output path
    representative_subunits_path = os.path.join(output_path, "_unified_representation", "assembly_output")
    transformations_path = os.path.join(output_path, "_unified_representation", "transformations")
    os.makedirs(representative_subunits_path, exist_ok=True)
    os.makedirs(transformations_path, exist_ok=True)

    print("--- Searching for subunits in supplied PDB files")
    pdb_path_to_partial_subunits = get_pdb_to_partial_subunits(pdbs_folder, subunits_info)

    print("--- Extracting representative subunits (for each subunit, its best scored model in the PDBs folder)")
    extract_representative_subunits(pdb_path_to_partial_subunits, subunits_info, representative_subunits_path)

    print("--- Extracting pairwise transformations between subunits (from each PDB file with 2 or more subunits)")
    extract_transformations(pdb_path_to_partial_subunits, subunits_info, representative_subunits_path,
                            transformations_path)

    print("--- Finished building unified representation")

    print("--- Running combinatorial assembly algorithm, may take a while")
    run_combfold(representative_subunits_path, subunits_info, transformations_path, crosslinks_path, output_path,
                 output_cif, max_results_number)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        run_on_pdbs_folder(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]))
    elif len(sys.argv) == 5:
        run_on_pdbs_folder(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]),
                           crosslinks_path=os.path.abspath(sys.argv[4]))
    else:
        print("usage: <script> subunits_info pdbs_folder output_path <optional: crosslinks.txt>")
