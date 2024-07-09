import re
from gensim.models import Word2Vec
from urllib.parse import urlparse
import fasttext
import numpy as np
import torch

# def tokenize_url(url):
#     parsed_url = urlparse(url)
#     words = []
#     words.append(parsed_url.scheme)
#     words.extend([word for word in parsed_url.netloc.split(".") if word])
#     words.extend([word for word in parsed_url.path.split("/") if word])
#     if parsed_url.query:
#         words.extend([word for word in parsed_url.query.split("&") if word])
#     return words


# ------------------------------- Word2Vec -------------------------------
def tokenize_url(url):
    return [token for token in re.split(r"[\./:?\-=]", url) if token]


def get_wv_model(urls):

    sentences = [tokenize_url(url) for url in urls]

    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    return model


def get_url_wv_embedding(model, url):
    words = tokenize_url(url)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return sum(word_vectors) / len(word_vectors) if word_vectors else None


# ------------------------------- FastText -------------------------------
def train_fasttext_model(urls):
    print(len(urls))
    with open("/home/jxlu/project/PhishHGMAE/embedding/data/corpus.txt", "w") as f:
        for url in urls:
            tokens = " ".join(tokenize_url(url))
            f.write(tokens + "\n")

    model = fasttext.train_unsupervised(
        "/home/jxlu/project/PhishHGMAE/embedding/data/corpus.txt",
        model="skipgram",
        dim=100,
    )
    return model


def get_url_fasttext_embedding(url, model):
    url_tokens = tokenize_url(url)
    url_embedding = [model.get_word_vector(token) for token in url_tokens]

    if len(url_embedding) == 0:
        return np.zeros(model.get_dimension())
    url_embedding = np.mean(url_embedding, axis=0)
    return url_embedding


def generate_fasttext_embeddings(urls, model):
    embeddings = [get_url_fasttext_embedding(url, model) for url in urls]
    return embeddings


# ------------------------------- Doc2Vec -------------------------------
def get_url_doc2vec_embedding(url, model):
    words = tokenize_url(url)
    return model.infer_vector(words)


def generate_doc2vec_embeddings(urls, model):
    embeddings = [get_url_doc2vec_embedding(url, model) for url in urls]
    return embeddings


# ------------------------------- BERT -------------------------------
def get_url_bert_embedding(url, tokenizer, model, device):
    words = tokenize_url(url)
    inputs = tokenizer(
        words,
        return_tensors="pt",
        padding=True,
        truncation=True,
        is_split_into_words=True,
    ).to(device)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    url_embedding = (
        torch.mean(last_hidden_states, dim=1).squeeze().detach().cpu().numpy()
    )
    return url_embedding


def generate_bert_embeddings(urls, tokenizer, model, device):
    embeddings = [get_url_bert_embedding(url, tokenizer, model, device) for url in urls]
    # return torch.tensor(embeddings)
    return embeddings
