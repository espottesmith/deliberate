from typing import Dict, List, Optional, Tuple, Collection, Callable

import networkx as nx
import numpy as np


def bucket_molecules(molecules: List[Dict], keys: List[Callable]):
    """
    Bucket molecules into nested dictionaries according to molecule properties
    specified in keys.

    The nested dictionary has keys as given in `keys`, and the innermost value is a
    list. For example, if `keys = ['formula', 'num_bonds', 'charge']`, then the returned
    bucket dictionary is something like:

    bucket[formula][num_bonds][charge] = [mol_entry1, mol_entry2, ...]

    where mol_entry1, mol_entry2, ... have the same formula, number of bonds, and charge.

    Args:
        entries: a list of molecule entries to bucket
        keys: each function should provide a molecule property

    Returns:
        Nested dictionary of molecule entry bucketed according to keys.
    """

    num_keys = len(keys)
    buckets = dict()  # type: ignore
    for m in molecules:
        b = buckets
        for i, j in enumerate(keys):
            v = j(m)
            if i == num_keys - 1:
                b.setdefault(v, []).append(m)
            else:
                b.setdefault(v, {})
            b = b[v]

    return buckets


def check_connectivity(mol: Dict) -> Tuple[bool, None]:
    """
    Check whether all atoms in a molecule is connected.

    Args:
        mol:
    Returns:
    """

    # all atoms are connected, not failing
    if nx.is_weakly_connected(mol["molecule_graph"].graph):
        return False, None

    # some atoms are not connected, failing
    else:
        return True, None


def check_species(mol: Dict, species: List[str]) -> Tuple[bool, str]:
    """
    Check whether molecule contains species given in `species`.

    Args:
        mol:
        species:

    Returns:
    """
    for s in species:
        if s in mol["species"]:
            return True, s
    return False, None


def check_bond_species(
    mol: Dict,
    bond_species: Collection[Tuple[str, str]],
):
    """
    Check whether molecule contains bonds with species specified in `bond_species`.
    """

    def get_bond_species(m):
        """
        Returns:
            A list of the two species associated with each bonds in the molecule.
        """
        res = list()
        for a1, a2 in m["bonds"]:
            s1 = m["species"][a1]
            s2 = m["species"][a2]
            res.append(sorted([s1, s2]))
        return res

    bond_species = [sorted(i) for i in bond_species]

    mol_bond_species = get_bond_species(mol)

    contains = False
    reason = list()
    for b in mol_bond_species:
        if b in bond_species:
            reason.append(b)
            contains = True

    return contains, reason


def check_bond_length(mol: Dict, bond_length_limit=None):
    """
    Check the length of bonds. If larger than allowed length, it fails.
    """

    def get_bond_lengths(m):
        """
        Returns:
            A list of tuple (species, length), where species are the two species
            associated with a bond and length is the corresponding bond length.
        """
        coords = m["molecule"].cart_coords
        res = list()
        for a1, a2 in m["bonds"]:
            s1 = m["species"][a1]
            s2 = m["species"][a2]
            c1 = np.asarray(coords[a1])
            c2 = np.asarray(coords[a2])
            length = np.linalg.norm(c1 - c2)
            res.append((tuple(sorted([s1, s2])), length))
        return res

    #
    # bond lengths references:
    # http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
    # https://slideplayer.com/slide/17256509/ page 29
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Chemical_Bonds/Bond_Lengths_and_Energies
    #
    # unit: Angstrom
    #

    if bond_length_limit is None:
        li_len = 2.8
        bond_length_limit = {
            # H
            ("H", "H"): 0.74,
            ("H", "H"): None,
            ("H", "C"): 1.09,
            ("H", "O"): 0.96,
            ("H", "F"): 0.92,
            ("H", "P"): 1.44,
            ("H", "N"): 1.01,
            ("H", "S"): 1.34,
            ("H", "Li"): li_len,
            # C
            ("C", "C"): 1.54,
            ("C", "O"): 1.43,
            ("C", "F"): 1.35,
            ("C", "P"): 1.84,
            ("C", "N"): 1.47,
            ("C", "S"): 1.81,
            ("C", "Li"): li_len,
            # O
            ("O", "O"): 1.48,
            ("O", "F"): 1.42,
            ("O", "P"): 1.63,
            ("O", "N"): 1.44,
            ("O", "S"): 1.51,
            ("O", "Li"): li_len,
            # F
            ("F", "F"): 1.42,
            ("F", "P"): 1.54,
            ("F", "N"): 1.39,
            ("F", "S"): 1.58,
            ("F", "Li"): li_len,
            # P
            ("P", "P"): 2.21,
            ("P", "N"): 1.77,
            ("P", "S"): 2.1,
            ("P", "Li"): li_len,
            # N
            ("N", "N"): 1.46,
            ("N", "S"): 1.68,
            ("N", "Li"): li_len,
            # S
            ("S", "S"): 2.04,
            ("S", "Li"): li_len,
            # Li
            ("Li", "Li"): li_len,
        }

        # multiply by 1.25 to relax the rule a bit
        tmp = dict()
        for k, v in bond_length_limit.items():
            if v is not None:
                v *= 1.25
            tmp[tuple(sorted(k))] = v
        bond_length_limit = tmp

    do_fail = False
    reason = list()

    bond_lengths = get_bond_lengths(mol)
    for b, length in bond_lengths:
        limit = bond_length_limit[b]
        if limit is not None and length > limit:
            reason.append("{}  {} ({})".format(b, length, limit))
            do_fail = True

    return do_fail, reason


def check_num_bonds(
    mol: Dict,
    allowed_connectivity: Optional[Dict[str, List[int]]] = None,
    exclude_species: Optional[List[str]] = None,
):
    """
    Check the number of bonds of each atom in a mol, without considering their bonding to
    metal element (e.g. Li), which forms coordinate bond with other atoms.

    If there are atoms violate the connectivity specified in allowed_connectivity,
    returns True.

    Args:
        mol:
        allowed_connectivity: {specie, [connectivity]}. Allowed connectivity by specie.
            If None, use internally defined connectivity.
        exclude_species: bond formed with species given in this list are ignored
            when counting the connectivity of an atom.
    """

    def get_neighbor_species(m):
        """
        Returns:
            A list of tuple (atom species, bonded atom species),
            where `bonded_atom_species` is a list.
            Each tuple represents an atom and its bonds.
        """
        res = [(s, list()) for s in m["species"]]
        for a1, a2 in m["bonds"]:
            s1 = m["species"][a1]
            s2 = m["species"][a2]
            res[a1][1].append(s2)
            res[a2][1].append(s1)
        return res

    if allowed_connectivity is None:
        allowed_connectivity = {
            "H": [1],
            "C": [1, 2, 3, 4],
            "O": [1, 2],
            "F": [1],
            "P": [1, 2, 3, 4, 5, 6],  # 6 for LiPF6
            "N": [1, 2, 3, 4, 5],
            "S": [1, 2, 3, 4, 5, 6],
            # metal
            "Li": [1, 2, 3],
        }

    exclude_species = ["Li"] if exclude_species is None else exclude_species

    neigh_species = get_neighbor_species(mol)

    do_fail = False
    reason = list()

    for a_s, n_s in neigh_species:
        num_bonds = len([s for s in n_s if s not in exclude_species])

        if num_bonds == 0:  # fine since we removed metal coordinate bonds
            continue

        if num_bonds not in allowed_connectivity[a_s]:
            reason.append("{} {}".format(a_s, num_bonds))
            do_fail = True

    return do_fail, reason


def filter_by_charge_and_isomorphism(
    molecules: List[Dict],
) -> List[Dict]:
    """
    For molecules of the same isomorphism and charge, remove the ones with higher electronic energies.

    Args:
        mol_entries: a list of molecule entries

    Returns:
        low_energy_entries: molecule entries with high free energy ones removed
    """

    # convert list of entries to nested dicts
    buckets = bucket_molecules(molecules, [
        lambda x: x["formula_alphabetical"],
        lambda x: len(x.get("bonds", list())),
        lambda x: x["charge"]
    ])

    all_entries = list()
    for formula in buckets:
        for num_bonds in buckets[formula]:
            for charge in buckets[formula][num_bonds]:

                # filter mols having the same formula, number bonds, and charge
                low_energy_entries = list()
                for entry in buckets[formula][num_bonds][charge]:

                    # try to find an entry_i with the same isomorphism to entry
                    idx = -1
                    for i, entry_i in enumerate(low_energy_entries):
                        if entry["molecule_graph"].isomorphic_to(entry_i["molecule_graph"]):
                            idx = i
                            break

                    if idx >= 0:
                        # entry has the same isomorphism as entry_i
                        if (
                            entry["thermo"]["raw"]["electronic_energy_Ha"]
                            < low_energy_entries[idx]["thermo"]["raw"]["electronic_energy_Ha"]
                        ):
                            low_energy_entries[idx] = entry

                    else:
                        # entry with a unique isomorphism
                        low_energy_entries.append(entry)

                all_entries.extend(low_energy_entries)

    return all_entries

def filter_connectivity(entries, verbose=False):
    """
    remove mols having atoms not connected to others
    """
    succeeded = list()
    for m in entries:
        fail, comment = check_connectivity(m)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_species(entries, not_allowed_species=None, verbose=False):
    """
    remove mols with specific species
    """
    if not_allowed_species is None:
        return entries

    succeeded = list()
    for m in entries:
        fail, comment = check_species(m, species=not_allowed_species)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_bond_species(entries, bond_species=(("Li", "H"), ("Li", "Li"), ("Mg", "Mg"), ("H", "Mg")), verbose=False):
    """
    remove mols with specific bond between species, e.g. Li-H
    """
    succeeded = list()
    for m in entries:
        fail, comment = check_bond_species(m, bond_species)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_bond_length(entries, verbose=False):
    """
    remove mols with larger bond length    
    """
    succeeded = list()
    for m in entries:
        fail, comment = check_bond_length(m)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_num_bonds(entries, verbose=False):
    """
    remove mols with unexpected number of bonds (e.g. more than 4 bonds for carbon),
    without considering metal species 
    """
    succeeded = list()
    for m in entries:
        fail, comment = check_num_bonds(m)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_dataset(
    entries: List[Dict],
    filters: List[Callable],
    bond_species: Optional[Collection[Tuple[str, str]]] = None,
    verbose=False
) -> List[Dict]:
    """
    Filter out some `bad` molecules. 
    """
    print("Number of starting molecules:", len(entries))

    remaining = entries
    for f in filters:
        if f.__name__ == "filter_bond_species":
            remaining = f(remaining, bond_species)
        else:
            remaining = f(remaining)
        print("Number of molecules after {}: {}".format(f.__name__, len(remaining)))

    return entries