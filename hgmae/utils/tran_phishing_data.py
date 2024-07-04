import pandas as pd
import numpy as np
import random


def trans_labels(data_dir, export_dir):

    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)

    node_labels = []

    for row in url_nodes.iterrows():
        if row[1]["category"] == "benign":
            node_labels.append(0)
        elif row[1]["category"] == "phishy":
            node_labels.append(1)

    label_array = np.array(node_labels, dtype="int32")

    label_npy_file = f"{export_dir}/labels.npy"

    np.save(label_npy_file, label_array)


# NOTE - Deprecated, please use data/neibor_phishing.py instead
def generate_nei_f(data_dir, export_dir):
    nei_f = [[] for _ in range(1018)]
    with open(f"{data_dir}/url_fqdn_relation_edges.csv") as f:
        url_fqdn_relation_edges = pd.read_csv(f)

    url_fqdn_relation_edges["url_id"] = url_fqdn_relation_edges["url_id"] - 1
    url_fqdn_relation_edges["fqdn_id"] = url_fqdn_relation_edges["fqdn_id"] - 1

    for row in url_fqdn_relation_edges.iterrows():
        if row[1]["fqdn_id"] not in nei_f[row[1]["url_id"]]:
            nei_f[row[1]["url_id"]].append(row[1]["fqdn_id"])
        # else:
        #     print("false")

    nei_f = [np.array(nei) for nei in nei_f]

    object_array = np.array(nei_f, dtype="object")

    npy_file = f"{export_dir}/nei_f.npy"

    np.save(npy_file, object_array)


def export_url_fqdn_edge_to_txt(data_dir, export_dir):
    with open(f"{data_dir}/url_fqdn_relation_edges.csv") as f:
        url_fqdn_relation_edges = pd.read_csv(f)

    url_fqdn_relation_edges["url_id"] = url_fqdn_relation_edges["url_id"] - 1
    url_fqdn_relation_edges["fqdn_id"] = url_fqdn_relation_edges["fqdn_id"] - 1

    txt_lines = []
    for row in url_fqdn_relation_edges.iterrows():
        txt_lines.append([row[1]["fqdn_id"], row[1]["url_id"]])

    txt_lines.sort()
    txt_lines = [f"{txt_line[0]}\t{txt_line[1]}\n" for txt_line in txt_lines]
    with open(f"{export_dir}/fu.txt", "w") as f:
        f.writelines(txt_lines)


def export_fqdn_registered_domain_edge_to_txt(data_dir, export_dir):
    with open(f"{data_dir}/fqdn_registered_domain_relation_edges.csv") as f:
        fqdn_registered_domain_relation_edges = pd.read_csv(f)

    fqdn_registered_domain_relation_edges["fqdn_id"] = (
        fqdn_registered_domain_relation_edges["fqdn_id"] - 1
    )
    fqdn_registered_domain_relation_edges["registered_domain_id"] = (
        fqdn_registered_domain_relation_edges["registered_domain_id"] - 1
    )

    txt_lines = []
    for row in fqdn_registered_domain_relation_edges.iterrows():
        txt_lines.append([row[1]["fqdn_id"], row[1]["registered_domain_id"]])

    txt_lines.sort()
    txt_lines = [f"{txt_line[0]}\t{txt_line[1]}\n" for txt_line in txt_lines]
    with open(f"{export_dir}/fr.txt", "w") as f:
        f.writelines(txt_lines)


def export_fqdn_ip_edge_to_txt(data_dir, export_dir):
    with open(f"{data_dir}/fqdn_ip_relation_edges.csv") as f:
        fqdn_ip_relation_edges = pd.read_csv(f)

    fqdn_ip_relation_edges["fqdn_id"] = fqdn_ip_relation_edges["fqdn_id"] - 1
    fqdn_ip_relation_edges["ip_id"] = fqdn_ip_relation_edges["ip_id"] - 1

    txt_lines = []
    for row in fqdn_ip_relation_edges.iterrows():
        txt_lines.append([row[1]["fqdn_id"], row[1]["ip_id"]])

    txt_lines.sort()
    txt_lines = [f"{txt_line[0]}\t{txt_line[1]}\n" for txt_line in txt_lines]
    with open(f"{export_dir}/fi.txt", "w") as f:
        f.writelines(txt_lines)


# Divide the data into training, testing andvalidation sets
def divide_datasets(data_dir, export_dir, ratio=[20, 40, 60], scale=0):
    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)
        url_nodes["url_id"] = url_nodes["url_id"] - 1

    benign_start_index = 0
    for node in url_nodes.iterrows():
        if node[1]["category"] == "phishy":
            benign_end_index = node[1]["url_id"] - 1
            break
    phishy_start_index = benign_end_index + 1
    phishy_end_index = url_nodes.iloc[-1]["url_id"]

    for num in ratio:
        all_nodes = list(range(phishy_end_index + 1))

        benign_random_index = random.sample(
            range(benign_start_index, benign_end_index + 1), num
        )
        phishy_random_index = random.sample(
            range(phishy_start_index, phishy_end_index + 1), num
        )
        random_index = sorted(benign_random_index + phishy_random_index)

        remain_nodes = [node for node in all_nodes if node not in random_index]
        test_index = random.sample(remain_nodes, int((scale * 2 - 200) / 2))

        remain_nodes = [node for node in remain_nodes if node not in test_index]
        val_index = random.sample(remain_nodes, int((scale * 2 - 200) / 2))

        random_index = np.array(random_index, dtype="int64")
        np.save(f"{export_dir}/train_{num}.npy", random_index)
        test_index = np.array(test_index, dtype="int64")
        np.save(f"{export_dir}/test_{num}.npy", test_index)
        val_index = np.array(val_index, dtype="int64")
        np.save(f"{export_dir}/val_{num}.npy", val_index)


if __name__ == "__main__":

    data_dir = "/home/jxlu/project/HGMAE/data/neo4j_mysql_export_1000"
    export_dir = "/home/jxlu/project/HGMAE/data/phishing_1000"

    trans_labels(data_dir, export_dir)

    # generate_nei_f(data_dir, export_dir)

    export_url_fqdn_edge_to_txt(data_dir, export_dir)

    export_fqdn_registered_domain_edge_to_txt(data_dir, export_dir)

    export_fqdn_ip_edge_to_txt(data_dir, export_dir)

    divide_datasets(data_dir, export_dir, scale=1000)
