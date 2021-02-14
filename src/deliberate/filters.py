from typing import Dict, List, Optional, Tuple, Collection, Callable

import networkx as nx
import numpy as np


def bucket_molecules(molecules: List[Dict], keys: List[Callable]) -> Dict:
    """
    Organize molecules into nested dictionaries according to molecule properties
    specified by the "keys".

    The nested dictionary has keys based on the functions used in "keys",
    and the innermost value is a list. For example, if
    `keys = [
        lambda x: x["formula_alphabetical"],
        lambda x: len(x.get("bonds", list())),
        lambda x: x["charge"]
    ]`,
    then the returned bucket dictionary is something like:

    bucket[formula][num_bonds][charge] = [mol_entry1, mol_entry2, ...]

    where mol_entry1, mol_entry2, ... have the same formula, number of bonds, and charge.

    Args:
        entries (List[Dict]): a list of molecule entries to bucket
        keys (List[Callable]): each function should return a property of the molecule

    Returns:
        Dict of molecule entry grouped according to keys.
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


def check_species(mol: Dict, species: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check whether molecule contains species given in `species`.

    Args:
        mol (Dict): A dictionary representing a molecule entry in LIBE
        species (List[str]): List of species to not allow in molecules

    Returns:
        Tuple[bool, Optional[str]]: if one of the disallowed species is included
        in this molecule, returns True and that specie; otherwise, returns
        False and None
    """
    for s in species:
        if s in mol["species"]:
            return True, s
    return False, None


def check_bond_species(
    mol: Dict,
    bond_species: Collection[Tuple[str, str]],
) -> Tuple[bool, Optional[List[Tuple[str, str]]]]:
    """
    Check whether particular types of bonds are present in the molecule.

    Args:
        mol (Dict): A dictionary representing a molecule entry in LIBE
        bond_species (Collection[Tuple[str, str]]): Collection of types
        of bonds (defined by the two species involved in the bond; e.g. 
        ("C", "O") for a C-O bond) not to be allowed in the molecule.

    Returns:
        Tuple[bool, Optional[List[Tuple[str, str]]]]: if any of the disallowed bonds
        is included in this molecule, returns True and those bond types; otherwise,
        returns False and None
    """

    def get_bond_species(m: Dict) -> List[List[str]]:
        """
        Args:
            m: A dictionary representing a molecule entry in LIBE

        Returns:
            A list of the types of bonds in the molecule, as defined by the species
            involved in the bond.
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

    if len(reason) == 0:
        return contains, None
    else:
        return contains, reason


def _get_bond_lengths(m):
    """
    Args:
        m: A dictionary representing a molecule entry in LIBE

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


def _get_neighbor_species(m):
    """
    Args:
        m: A dictionary representing a molecule entry in LIBE

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


def check_bond_length(mol: Dict) -> Tuple[bool, Optional[List[str]]]:
    """
    Check the length of bonds. If larger than allowed length, it fails.

    The allowed lengths are based on reference values (for instance, see
    http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf).
    A bond with length greater than 125% of these values is considered to be in
    violation.

    The chosen "standard" bond length for all Li coordinate bonds is taken to be 2.8
    Angstrom (before the allowed 25% increase).

    Args:
        mol (Dict): A dictionary representing a molecule entry in LIBE

    Returns:
        Tuple[bool, Optional[List[str]]]: if any bonds are too long, returns True
        and a formatted string indicating the type of bond, the actual length, and
        the maximum allowed; otherwise, returns False and None
    """

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

    bond_lengths = _get_bond_lengths(mol)
    for b, length in bond_lengths:
        limit = bond_length_limit[b]
        if limit is not None and length > limit:
            reason.append("{}  {} ({})".format(b, length, limit))
            do_fail = True

    return do_fail, reason


def check_num_bonds(
    mol: Dict,
    allowed_connectivity: Optional[Dict[str, List[int]]] = None,
) -> Tuple[bool, Optional[List[str]]]:
    """
    Check the number of bonds of each atom in a mol (ignoring metal coordinate bonding)

    If there are atoms violate the connectivity specified in allowed_connectivity,
    returns True.

    Args:
        mol (Dict): A dictionary representing a molecule entry in LIBE
        allowed_connectivity (Optional[Dict[str, List[int]]]): Dict of format
        {specie: [conn_1, conn_2, ...]}. Allowed connectivity by specie. As an example,
        if C was allowed to make 1, 2, 3, or 4 bonds, then one would include
        {"C": [1, 2, 3, 4]}. If None, use internally defined connectivity.

    Returns:
        Tuple[bool, Optional[List[str]]]: if any atom has an inappropriate number of
        bonds, returns True and a formatted string indicating the specie and the number
        of bonds it has; otherwise, returns False and None
    """

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

    exclude_species = ["Li"]

    neigh_species = _get_neighbor_species(mol)

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
    For molecules of the same isomorphism and charge, remove the ones with higher
    electronic energies.

    Args:
        mol_entries (List[Dict]): a list of dictionaries representing molecule
        entries in LIBE

    Returns:
        low_energy_entries (List[Dict]): molecule entries with high free energy
        ones removed
    """

    # convert list of entries to nested dicts
    buckets = bucket_molecules(
        molecules, [lambda x: x["formula_alphabetical"], lambda x: len(x.get("bonds", list())), lambda x: x["charge"]]
    )

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


def filter_species(
    molecules: List[Dict], not_allowed_species: Optional[List[str]] = None, verbose: bool = False
) -> List[Dict]:
    """
    Remove molecules with specific species

    Args:
        molecules (List[Dict]): a list of dictionaries representing molecule entries in LIBE
        not_allowed_species (Optional[List[str]]): List of species not allowed
        verbose (bool): If True (default False), indicate which molecules failed this check

    Returns:
        succeeded (List[Dict]): Molecule entry dictionaries that passed this check
    """
    if not_allowed_species is None:
        return molecules

    succeeded = list()
    for m in molecules:
        fail, comment = check_species(m, species=not_allowed_species)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_bond_species(
    molecules: List[Dict],
    bond_species: Collection[Tuple[str, str]] = (("Li", "H"), ("Li", "Li")),
    verbose: bool = False,
) -> List[Dict]:
    """
    Remove molecules with specific bond between species, e.g. Li-Li

    Args:
        molecules (List[Dict]): a list of dictionaries representing molecule entries
        in LIBE
        bond_species (Collection[Tuple[str, str]]): Collection of types
        of bonds (defined by the two species involved in the bond; e.g. 
        ("C", "O") for a C-O bond) not to be allowed in the molecule.
        verbose (bool): If True (default False), indicate which molecules failed
        this check

    Returns:
        succeeded (List[Dict]): Molecule entry dictionaries that passed this check
    """
    succeeded = list()
    for m in molecules:
        fail, comment = check_bond_species(m, bond_species)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_bond_length(molecules: List[Dict], verbose: bool = False) -> List[Dict]:
    """
    remove mols with larger bond length

    Args:
        molecules (List[Dict]): a list of dictionaries representing molecule entries in LIBE
        verbose (bool): If True (default False), indicate which molecules failed this check

    Returns:
        succeeded (List[Dict]): Molecule entry dictionaries that passed this check
    """
    succeeded = list()
    for m in molecules:
        fail, comment = check_bond_length(m)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_num_bonds(molecules: List[Dict], verbose: bool = False) -> List[Dict]:
    """
    remove mols with unexpected number of bonds (e.g. more than 4 bonds for carbon),
    without considering metal species

    Args:
        molecules (List[Dict]): a list of dictionaries representing molecule entries in LIBE
        verbose (bool): If True (default False), indicate which molecules failed this check

    Returns:
        succeeded (List[Dict]): Molecule entry dictionaries that passed this check
    """
    succeeded = list()
    for m in molecules:
        fail, comment = check_num_bonds(m)
        if fail:
            if verbose:
                print(m["molecule_id"], comment)
        else:
            succeeded.append(m)

    return succeeded


def filter_dataset(
    molecules: List[Dict],
    filters: List[Callable],
    verbose=False,
) -> List[Dict]:
    """
    Filter out `bad` molecules.

    Args:
        molecules (List[Dict]): a list of dictionaries representing molecule entries in LIBE
        filters (List[Callable]): list of filter functions (such as those included in this module)
        bond_species (Optional[Collection[Tuple[str, str]]]):
        verbose (bool): If True (default False), indicate which molecules failed and why

    Returns:
        molecules (List[Dict]): Molecule entry dictionaries that passed this check
    """
    print("Number of starting molecules:", len(molecules))

    remaining = molecules
    for f in filters:
        remaining = f(remaining)
        print("Number of molecules after {}: {}".format(f.__name__, len(remaining)))

    return remaining
