# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import os
import copy
from itertools import combinations_with_replacement
import time
from pathlib import Path

import numpy as np

from monty.serialization import dumpfn, loadfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender


def combine_mol_graphs(molgraph_1: MoleculeGraph, molgraph_2: MoleculeGraph) -> MoleculeGraph:
    """
    Create a combined MoleculeGraph based on two initial MoleculeGraphs.

    Args:
        molgraph_1 (MoleculeGraph)
        molgraph_2 (MoleculeGraph)

    Returns:
        copy_1 (MoleculeGraph)
    """
    # This isn't strictly necessary, but we center both molecules and shift the second
    # For 3D structure generation, having the two molecules appropriately separated is
    # helpful

    radius_1 = np.amax(molgraph_1.molecule.distance_matrix)
    radius_2 = np.amax(molgraph_2.molecule.distance_matrix)

    copy_1 = copy.deepcopy(molgraph_1)
    copy_1.molecule.translate_sites(list(range(len(molgraph_1.molecule))), -1 * molgraph_1.molecule.center_of_mass)

    copy_2 = copy.deepcopy(molgraph_2)
    copy_2.molecule.translate_sites(
        list(range(len(copy_2.molecule))),
        -1 * copy_2.molecule.center_of_mass + np.array([radius_1 + radius_2 + 1.0, 0.0, 0.0]),
    )

    for site in copy_2.molecule:
        copy_1.insert_node(len(copy_1.molecule), site.specie, site.coords)

    for edge in copy_2.graph.edges():
        side_1 = edge[0] + len(molgraph_1.molecule)
        side_2 = edge[1] + len(molgraph_1.molecule)
        copy_1.add_edge(side_1, side_2)

    copy_1.molecule.set_charge_and_spin(molgraph_1.molecule.charge + molgraph_2.molecule.charge)

    return copy_1


def identify_connectable_heavy_atoms(mol_graphs: List[MoleculeGraph]) -> List[List[int]]:
    """
    Identify the heavy atoms in a molecule that can form additional bonds,
    based on valence rules

    Args:
        mol_graphs (List[MoleculeGraph]): List of initial (fragment) MoleculeGraphs

    Returns:
        heavy_atoms_index_list (List[List[int]]): List of the appropriate indices for
        each molecule graph in mol_graphs.
    """

    bond_max = {"C": 4, "P": 6, "S": 6, "O": 2, "N": 3, "F": 1}

    heavy_atoms_index_list = list()
    for i, mol_graph in enumerate(mol_graphs):
        heavy_atoms_in_mol = list()
        num_atoms = len(mol_graph.molecule)
        for j in range(num_atoms):
            connected_sites = mol_graph.get_connected_sites(j)
            num_connected_sites = len(connected_sites)
            element = str(mol_graph.molecule[j].specie)

            if element in ["Li", "H"]:
                if num_connected_sites == 0 and num_atoms == 1:
                    heavy_atoms_in_mol.append(j)

            else:
                metal_count = 0

                for k, site in enumerate(connected_sites):
                    if str(site.site.specie) == "Li":
                        metal_count += 1

                if num_connected_sites - metal_count < bond_max[element]:
                    heavy_atoms_in_mol.append(j)

        heavy_atoms_index_list.append(heavy_atoms_in_mol)

    return heavy_atoms_index_list


def generate_combinations(mol_graphs: List[MoleculeGraph], directory: Path, max_size: Optional[int] = None):
    """
    Generate all combination of molecule/atom indices that can participate in recombination
    by looping through all molecule pairs(including a mol and itself) and all connectable
    heavy atoms in each molecule.

    Args:
        mol_graphs (List[MoleculeGraph]): List of initial (fragment) MoleculeGraphs
        directory (Path): Path in which to place the output files
        max_size (Optional[int]): If not None (default), only recombinant molecules
        with less than this number of electrons will be allowed.

    Returns:
        final_list (List[MoleculeGraph]): List of all generated recombinant molecules
    """

    combinations_file = directory / "combinations.txt"
    mol_graphs_file = directory / "mol_graphs_recombination.json"

    with open(combinations_file.as_posix(), "w") as combos:
        combos.write("mol_1\tatom_1\tmol_2\tatom_2\n")
        final_list = list()
        heavy_atoms_index_list = identify_connectable_heavy_atoms(mol_graphs)
        num_mols = len(mol_graphs)
        all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))
        for pair_index in all_mol_pair_index:
            mol_graph1 = mol_graphs[pair_index[0]]
            mol_graph2 = mol_graphs[pair_index[1]]

            total_charge = mol_graph1.molecule.charge + mol_graph2.molecule.charge
            total_electrons = mol_graph1.molecule._nelectrons + mol_graph2.molecule._nelectrons

            if int(total_charge) not in {-1, 0, 1}:
                continue

            if max_size is not None:
                if total_electrons > max_size:
                    continue

            heavy_atoms_1 = heavy_atoms_index_list[pair_index[0]]
            heavy_atoms_2 = heavy_atoms_index_list[pair_index[1]]
            if len(heavy_atoms_1) == 0 or len(heavy_atoms_2) == 0:
                continue
            else:
                for i, atom1 in enumerate(heavy_atoms_1):
                    for j, atom2 in enumerate(heavy_atoms_2):
                        specie1 = str(mol_graph1.molecule[atom1].specie)
                        specie2 = str(mol_graph2.molecule[atom2].specie)

                        if specie1 == "Li" and specie2 == "Li":
                            continue

                        combined_mol_graph = combine_mol_graphs(mol_graph1, mol_graph2)
                        combined_mol_graph.add_edge(atom1, atom2 + len(mol_graph1.molecule))

                        match = False
                        for entry in mol_graphs:
                            if (
                                combined_mol_graph.isomorphic_to(entry)
                                and combined_mol_graph.molecule.charge == entry.molecule.charge
                            ):
                                match = True
                                break
                        if match:
                            continue

                        index = None
                        for ii, mol_graph in enumerate(final_list):
                            if (
                                mol_graph.isomorphic_to(combined_mol_graph)
                                and combined_mol_graph.molecule.charge == entry.molecule.charge
                            ):
                                index = ii
                                break
                        if index is None:
                            index = len(final_list)
                            final_list.append(combined_mol_graph)

                        combos.write("{}\t{}\t{}\t{}\t{}\n".format(pair_index[0], atom1, pair_index[1], atom2, index))

    dumpfn(final_list, mol_graphs_file.as_posix())

    return final_list


def parse_combinations_file(filepath: Path) -> List[Tuple[int, int, int, int, int]]:
    """
    Parse a text file to extract reaction information.

    Args:
       filepath (Path)

    Return:
        reactions (List[Tuple[int, int, int, int, int]]): List of reactions. The five elements
            in each reaction are, in order:
                molecule_1_index
                atom_1_index
                molecule_2_index
                atom_2_index
                product_index

    """
    with open(filepath.as_posix()) as combo_file:
        lines = combo_file.readlines()

        reactions = list()
        # Skip first line - header
        for line in lines[1:]:
            line_parsed = [int(x) for x in line.strip().split("\t")]
            reactions.append(tuple(line_parsed))

        return reactions


def generate_bondnet_files(
    orig_mol_graphs: List[MoleculeGraph],
    recombinant_mol_graphs: List[MoleculeGraph],
    combinations: List[Tuple[int, int, int, int, int]],
    output_directory: Path,
):
    """
    Generate input files for BonDNet

    Args:
        orig_mol_graphs (List[MoleculeGraph]): List of molecule graphs used to generate the recombinant molecules
        recombinant_mol_graphs (List[MoleculeGraph]): List of recombinant molecule graphs
        combinations (List[Tuple[int, int, int, int, int]]): List of reaction tuples of format
                molecule_1_index
                atom_1_index
                molecule_2_index
                atom_2_index
                product_index
        output_directory (Path)

    Returns:
        None

    """
    total_mol_graphs = orig_mol_graphs + recombinant_mol_graphs

    bookmark = len(orig_mol_graphs)

    with open((output_directory / "reactions.csv").as_posix(), "w") as rxn_file:
        rxn_file.write("reactant,product1,product2\n")
        for ii, combination in enumerate(combinations):
            rxn_file.write("{},{},{}\n".format(bookmark + combination[4], combination[0], combination[2]))

    dumpfn(total_mol_graphs, (output_directory / "mol_graphs.json").as_posix())
