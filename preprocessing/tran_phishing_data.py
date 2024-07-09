import pandas as pd
import numpy as np
import random
import scipy.sparse as sp


# Save url label to npy file
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


# Export url_fqdn_relation_edges to txt file
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


# Export fqdn_registered_domain_relation_edges to txt file
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


# Export fqdn_ip_relation_edges to txt file
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


# Export the neighbor fqdns for every url.
# NOTE - Deprecated, please use data/neibor_phishing.py instead
# def generate_nei_f(data_dir, export_dir):
#     nei_f = [[] for _ in range(1018)]
#     with open(f"{data_dir}/url_fqdn_relation_edges.csv") as f:
#         url_fqdn_relation_edges = pd.read_csv(f)

#     url_fqdn_relation_edges["url_id"] = url_fqdn_relation_edges["url_id"] - 1
#     url_fqdn_relation_edges["fqdn_id"] = url_fqdn_relation_edges["fqdn_id"] - 1

#     for row in url_fqdn_relation_edges.iterrows():
#         if row[1]["fqdn_id"] not in nei_f[row[1]["url_id"]]:
#             nei_f[row[1]["url_id"]].append(row[1]["fqdn_id"])

#     nei_f = [np.array(nei) for nei in nei_f]

#     object_array = np.array(nei_f, dtype="object")

#     npy_file = f"{export_dir}/nei_f.npy"

#     np.save(npy_file, object_array)


def generate_nei_f(export_dir):
    fu = np.genfromtxt(f"{export_dir}/fu.txt")
    u_n = {}  # url neighbor nodes
    for i in fu:
        if i[1] not in u_n:
            u_n[int(i[1])] = []
            u_n[int(i[1])].append(int(i[0]))
        else:
            u_n[int(i[1])].append(int(i[0]))
    keys = sorted(u_n.keys())
    u_n = [u_n[i] for i in keys]
    u_n = [np.array(nei) for nei in u_n]
    u_n = np.array(u_n, dtype="object")
    np.save(f"{export_dir}/nei_f.npy", u_n)


# Divide the data into training, testing and validation sets
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


def generate_metapath_adjacency(data_dir, export_dir):

    fu = np.genfromtxt(f"{export_dir}/fu.txt")
    fr = np.genfromtxt(f"{export_dir}/fr.txt")
    fi = np.genfromtxt(f"{export_dir}/fi.txt")

    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)
        U = url_nodes.iloc[-1]["url_id"]

    with open(f"{data_dir}/fqdn_nodes.csv") as f:
        fqdn_nodes = pd.read_csv(f)
        F = fqdn_nodes.iloc[-1]["fqdn_id"]

    with open(f"{data_dir}/registered_domain_nodes.csv") as f:
        registered_domain_nodes = pd.read_csv(f)
        R = registered_domain_nodes.iloc[-1]["registered_domain_id"]

    with open(f"{data_dir}/ip_nodes.csv") as f:
        ip_nodes = pd.read_csv(f)
        I = ip_nodes.iloc[-1]["ip_id"]

    print(U, F, R, I)

    fu_ = sp.coo_matrix(
        (np.ones(fu.shape[0]), (fu[:, 0], fu[:, 1])), shape=(F, U)
    ).toarray()
    fr_ = sp.coo_matrix(
        (np.ones(fr.shape[0]), (fr[:, 0], fr[:, 1])), shape=(F, R)
    ).toarray()
    fi_ = sp.coo_matrix(
        (np.ones(fi.shape[0]), (fi[:, 0], fi[:, 1])), shape=(F, I)
    ).toarray()

    ufu = np.matmul(fu_.T, fu_) > 0
    ufu = sp.coo_matrix(ufu)
    sp.save_npz(f"{export_dir}/ufu.npz", ufu)

    ufr = np.matmul(fu_.T, fr_) > 0
    ufrfu = np.matmul(ufr, ufr.T) > 0
    ufrfu = sp.coo_matrix(ufrfu)
    sp.save_npz(f"{export_dir}/ufrfu.npz", ufrfu)

    ufi = np.matmul(fu_.T, fi_) > 0
    ufifu = np.matmul(ufi, ufi.T) > 0
    ufifu = sp.coo_matrix(ufifu)
    sp.save_npz(f"{export_dir}/ufifu.npz", ufifu)


if __name__ == "__main__":

    data_dir = "/home/jxlu/project/PhishHGMAE/data/neo4j_mysql_export_1000"
    export_dir = "/home/jxlu/project/PhishHGMAE/data/phishing_1000"

    trans_labels(data_dir, export_dir)

    export_url_fqdn_edge_to_txt(data_dir, export_dir)

    export_fqdn_registered_domain_edge_to_txt(data_dir, export_dir)

    export_fqdn_ip_edge_to_txt(data_dir, export_dir)

    generate_nei_f(export_dir)

    divide_datasets(data_dir, export_dir, scale=1000)

    generate_metapath_adjacency(data_dir, export_dir)