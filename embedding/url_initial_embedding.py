import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertTokenizer, BertModel
import torch

# -------------------- Word2Vec --------------------

# if __name__ == "__main__":

#     urls = [
#         "https://example.com/path/to/page?query=1",
#         "https://anotherexample.com/different/path?param=value",
#     ]

#     data_dir = "/home/jxlu/project/HGMAE/data/neo4j_mysql_export_1000"
#     export_dir = "/home/jxlu/project/PhishHGMAE/data/phishing_1000"

#     with open(f"{data_dir}/url_nodes.csv") as f:
#         url_nodes = pd.read_csv(f)

#     urls = [url for url in url_nodes["url"]]

#     # print(urls)
#     model = get_wv_model(urls)

#     url_embeddings = []
#     for url in tqdm(urls):
#         url_embedding = get_url_wv_embedding(model, url)
#         url_embeddings.append(url_embedding)

#     data_array = np.array(url_embeddings, dtype="float32")

#     # np.save(f"{export_dir}/u_wv_feat.npy", data_array)

#     print(type(url_embeddings))

# -------------------- FastText --------------------

# if __name__ == "__main__":

#     data_dir = "/home/jxlu/project/PhishHGMAE/embedding/data"
#     export_dir = "/home/jxlu/project/PhishHGMAE/data/phishing_1000"

#     with open(f"{data_dir}/url_nodes.csv") as f:
#         url_nodes = pd.read_csv(f)

#     urls = [url for url in url_nodes["url"]]

#     with open(
#         "/home/jxlu/project/PhishHGMAE/data/phishy_urls/phishy_benign_urls.csv", "r"
#     ) as f:
#         train_urls = f.readlines()
#         train_urls = [url.strip("\n") for url in train_urls]
#     model = train_fasttext_model(train_urls)

#     embeddings = generate_fasttext_embeddings(urls, model)

#     data_array = np.array(embeddings, dtype="float32")
#     np.save(f"{export_dir}/u_fasttext_feat.npy", data_array)

# -------------------- Doc2Vec --------------------
# if __name__ == "__main__":

#     data_dir = "/home/jxlu/project/PhishHGMAE/embedding/data"
#     export_dir = "/home/jxlu/project/PhishHGMAE/data/phishing_1000"

#     with open(f"{data_dir}/url_nodes.csv") as f:
#         url_nodes = pd.read_csv(f)
#     urls = [url for url in url_nodes["url"]]

#     documents = [
#         TaggedDocument(words=tokenize_url(url), tags=[str(i)])
#         for i, url in enumerate(urls)
#     ]

#     model = Doc2Vec(
#         documents, vector_size=100, window=2, min_count=1, workers=4, epochs=100
#     )

#     embeddings = generate_doc2vec_embeddings(urls, model)

#     data_array = np.array(embeddings, dtype="float32")
#     np.save(f"{export_dir}/u_doc2vec_feat.npy", data_array)

# -------------------- BERT --------------------
if __name__ == "__main__":

    data_dir = "/home/jxlu/project/PhishHGMAE/embedding/data"
    export_dir = "/home/jxlu/project/PhishHGMAE/data/phishing_1000"

    with open(f"{data_dir}/url_nodes.csv") as f:
        url_nodes = pd.read_csv(f)
    urls = [url for url in url_nodes["url"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)

    embeddings = generate_bert_embeddings(urls, tokenizer, model, device)
    data_array = np.array(embeddings, dtype="float32")
    np.save(f"{export_dir}/u_bert_feat.npy", data_array)
