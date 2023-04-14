import os
import argparse
import json

TEST_DIR = "mol2_test"
SUPPORTED_NODE_TYPES = ["atom", "aa"]


def parse_mol2(mol2_file):
    """
    Read a .mol2 file, build a structured data structure.
    """

    status = "none"
    parsed_mol2 = {"node": {}, "bond": {}}
    for line in mol2_file.readlines():
        if line == "@<TRIPOS>ATOM\n":
            status = "atom"
            continue
        elif line == "@<TRIPOS>BOND\n":
            status = "bond"
            continue
        elif line == "@<TRIPOS>UNITY_ATOM_ATTR\n":
            status = "continue"
            continue
            
        if status == "bond" and line.startswith("@<TRIPOS>"):
            return parsed_mol2

        if status == "atom":
            atom_id, atom_name, x, y, z, atom_type, subst_id, subst_name = line.split()[
                :8]
            parsed_mol2["node"][atom_id] = {
                "atom_name": atom_name,
                "subst_id": subst_id,
                "subst_name": subst_name
            }
        elif status == "bond":
            bond_id, origin_atom_id, target_atom_id, bond_type = line.split()
            parsed_mol2["bond"][bond_id] = {
                "origin_atom_id": origin_atom_id,
                "target_atom_id": target_atom_id,
                "bond_type": bond_type
            }
        elif status == "continue":
            continue
    return parsed_mol2


def extend_node(parsed_mol2, node_type):
    """
    Number the bond, transforming an edge into a node.
    """

    if node_type == "atom":
        atom_ids = list(parsed_mol2["node"].keys())
        max_node_id = max(map(lambda x: int(x), atom_ids))
    elif node_type == "aa":
        # Concatenate subst_id and subst_name, use it as a new subst_name
        concatenated_subst_id_set = set()
        for atom_id in parsed_mol2["node"]:
            concatenated_subst_id = parsed_mol2["node"][atom_id]["subst_id"] + \
                '-' + parsed_mol2["node"][atom_id]["subst_name"]
            # concatenated_subst_id = parsed_mol2["node"][atom_id]["subst_name"][0:3]
            parsed_mol2["node"][atom_id]["subst_name"] = concatenated_subst_id
            concatenated_subst_id_set.add(concatenated_subst_id)
        unique_subst_id_list = sorted(list(concatenated_subst_id_set))
        for atom_id in parsed_mol2["node"]:
            subst_name = parsed_mol2["node"][atom_id]["subst_name"]
            parsed_mol2["node"][atom_id]["subst_id"] = unique_subst_id_list.index(
                subst_name) + 1
        # Get the maximum subst_id
        subst_ids = list(
            map(lambda x: x["subst_id"], parsed_mol2["node"].values()))
        max_node_id = max(subst_ids)
    else:
        return None

    for bond_id in parsed_mol2["bond"].keys():
        parsed_mol2["bond"][bond_id]["fake_node_id"] = int(
            bond_id) + max_node_id
    return parsed_mol2


def build_graph(extended_mol2, node_type, allow_ring):
    graph = {
        "edges": [],
        "features": {}
    }
    for bond in extended_mol2["bond"].values():
        origin_atom_id, target_atom_id, bond_type, fake_node_id = bond.values()
        ignore_edge = False
        ignore_fake_node = False

        if node_type == "atom":
            origin_node_id = origin_atom_id
            target_node_id = target_atom_id
            origin_node_feature = extended_mol2["node"][origin_atom_id]["atom_name"]
            target_node_feature = extended_mol2["node"][target_atom_id]["atom_name"]
        elif node_type == "aa":
            origin_node_id = extended_mol2["node"][origin_atom_id]["subst_id"]
            target_node_id = extended_mol2["node"][target_atom_id]["subst_id"]
            origin_node_feature = extended_mol2["node"][origin_atom_id]["subst_name"]
            target_node_feature = extended_mol2["node"][target_atom_id]["subst_name"]
            if origin_node_id == target_node_id and not allow_ring:
                ignore_edge = True
                ignore_fake_node = True
        else:
            return None

        fake_node_id = str(fake_node_id)
        if not ignore_edge:
            graph["edges"].append([int(origin_node_id), int(fake_node_id)])
            graph["edges"].append([int(fake_node_id), int(target_node_id)])
        if not ignore_fake_node:
            graph["features"][fake_node_id] = bond_type
        graph["features"][origin_node_id] = origin_node_feature
        graph["features"][target_node_id] = target_node_feature
    return graph


def main(input_dir, output_dir, node_type, allow_ring):
    # working_dir = TEST_DIR
    if node_type not in SUPPORTED_NODE_TYPES:
        print("Unknown node type: '{}'".format(node_type))
        exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = os.listdir(input_dir)
    for mol2_filename in file_list:
        if os.path.splitext(mol2_filename)[1] != ".mol2":
            continue
        print(mol2_filename)
        with open(os.path.join(input_dir, mol2_filename)) as f:
            parsed_mol2 = parse_mol2(f)
        extend_node(parsed_mol2, node_type)
        graph = build_graph(parsed_mol2, node_type, allow_ring)
        output_filename = os.path.splitext(mol2_filename)[0] + ".json"
        with open(os.path.join(output_dir, output_filename), "w") as outf:
            json.dump(graph, outf)
            # outf.write(str(graph))
        # print(graph)
        # return graph


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Build graphs from mol2 files.")
    arg_parser.add_argument(
        "-i", "--input", help="mol2 file input dir", required=True)
    arg_parser.add_argument(
        "-o", "--output", help="graph file output dir", required=True)
    arg_parser.add_argument(
        "-n", "--nodetype", help="[atom, aa] Node type, default:'aa'", default="aa")
    arg_parser.add_argument(
        "-r", "--allowring", help="if this flag is set, self ring edge is allowed (only available when NODETYPE == aa)", action="store_true")
    args = arg_parser.parse_args()
    main(args.input, args.output, args.nodetype, args.allowring)
