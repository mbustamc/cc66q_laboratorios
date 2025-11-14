# -*- coding: utf-8 -*-
import pandas as pd

dataset_df = pd.read_csv("https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/new/assignment_1/train/train.tsv", sep="\t")

dataset_df.sample(5)

dataset_df["clase"].value_counts()

target_classes = list(dataset_df['clase'].unique())
target_classes


import nltk
import numpy as np

from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin


import matplotlib.pyplot as plt

# word2vec
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phrases, Phraser

# Pytorch imports
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc

from torch.optim import Adam

### Crear un clasificador basado en RNN

from sklearn.model_selection import train_test_split

def split_df(df, test_size = 0.2, seed: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['clase'])
    return train_df, test_df


def load_df(df):
   for _, row in df.iterrows():
     yield row['texto'], row['clase']


train_df, test_df = split_df(dataset_df)

"""#### Definir el vocabulario."""

import re

def tokenizer_es(text: str):
    return re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", str(text).lower())

def normalize(text):
    text = re.sub(r"http\S+","<URL>", text)
    text = re.sub(r"@\w+","<USER>", text)
    text = re.sub(r"#(\w+)","\\1", text)     # quita # pero deja la palabra
    return text

def tokenizer_es_norm(t):
    return tokenizer_es(normalize(t))


tokenizer = get_tokenizer("basic_english")

def generate_tokens(example: iter):
    for text, _ in example:
        yield tokenizer_es_norm(text)



def build_vocab(example: iter, min_freq=1):
    vocab = build_vocab_from_iterator(
        generate_tokens(example),
        min_freq=min_freq,
        specials=["<PAD>", "<UNK>"]   # <--- añade PAD primero
    )
    vocab.set_default_index(vocab["<UNK>"])
    return vocab





train_data = load_df(train_df)
vocab = build_vocab(train_data)

"""#### Cargar el `DataLoader`

Recuerde que podría necesitar una función intermedia para procesar cada batch durante el entrenamiento, pero no es obligatorio hacerlo.
"""

train_dataset = load_df(train_df)
test_dataset = load_df(test_df)
train_dataset, test_dataset  = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)

max_words = 25

MAX_WORDS = 50
PAD_ID = vocab["<PAD>"]  # ahora sí existe
label_list = target_classes
label2idx = {lbl: i for i, lbl in enumerate(label_list)}

'''def vectorize_batch(batch):
    # batch: lista de (texto, etiqueta)
    X_texts, Y_labels = zip(*batch)                     # <-- orden correcto

    # 1) tokenizar
    tokenized = [tokenizer_es_norm(t) for t in X_texts]      # lista[list[str]]

    # 2) tokens -> ids con torchtext
    ids = [vocab.lookup_indices(toks) for toks in tokenized]  # lista[list[int]]

    # 3) padding / truncado
    seq_len = min(MAX_WORDS, max(len(s) for s in ids)) if ids else MAX_WORDS
    padded = [
        (s[:seq_len] + [PAD_ID] * (seq_len - len(s))) if len(s) < seq_len else s[:seq_len]
        for s in ids
    ]

    # 4) tensores (Embedding espera long)
    X = torch.tensor(padded, dtype=torch.long)

    # 5) etiquetas string -> id (no restes 1)
    y = torch.tensor([label2idx[lbl] for lbl in Y_labels], dtype=torch.long)

    return X, y'''


'''def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X] ## Bringing all samples to max_words length.

    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]'''



MAX_WORDS = 100
PAD_ID = vocab["<PAD>"]

def vectorize_batch(batch):
    X_texts, Y_labels = zip(*batch)
    tokenized = [tokenizer_es_norm(t) for t in X_texts]
    ids = [vocab.lookup_indices(toks) for toks in tokenized]
    lengths = torch.tensor([min(len(s), MAX_WORDS) for s in ids], dtype=torch.long)
    T = min(MAX_WORDS, max(lengths).item())
    padded = [(s[:T] + [PAD_ID]*(T-len(s))) if len(s)<T else s[:T] for s in ids]
    X = torch.tensor(padded, dtype=torch.long)
    y = torch.tensor([label2idx[l] for l in Y_labels], dtype=torch.long)
    return X, y, lengths


train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=vectorize_batch, shuffle=True)
test_loader  = DataLoader(test_dataset , batch_size=64, collate_fn=vectorize_batch)

"""#### Definir la red recurrente.

Recuerde que debe difirnir los hyperparametros que estime conveniente.

# OJO ____ ACA NO SE DEBE COPIAR. YO LO HICE PARA PROBAR
"""

class RNNClassifier(nn.Module):
    def __init__(self):
        super(RNNClassifier, self).__init__()
        pass
    def forward(self, X_batch):
        pass

"""# La siguiente es la que copie!"""

embed_len = 50
hidden_dim = 50
n_layers=1

class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(len(vocab), embed_len, padding_idx=PAD_ID)
        self.rnn = nn.GRU(input_size=embed_len, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_dim*2, len(target_classes))  # *2 por bidireccional

    def forward(self, X_batch, lengths):
        emb = self.embedding_layer(X_batch)  # (B, T, E)
        # pack (importante: lengths en CPU y ordenadas desc si enforce_sorted=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.rnn(packed)   # h_n: (num_layers*2, B, H)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # concat direcciones
        logits = self.linear(self.dropout(h_last))
        return logits

embed_len = 50
hidden_dim1 = 50
hidden_dim2 = 60
hidden_dim3 = 75
n_layers=1

class StackingRNNClassifier(nn.Module):
    def __init__(self):
        super(StackingRNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.rnn1 = nn.RNN(input_size=embed_len, hidden_size=hidden_dim1, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(input_size=hidden_dim1, hidden_size=hidden_dim2, num_layers=1, batch_first=True)
        self.rnn3 = nn.RNN(input_size=hidden_dim2, hidden_size=hidden_dim3, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim3, len(target_classes))

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn1(embeddings, torch.randn(n_layers, len(X_batch), hidden_dim1))
        output, hidden = self.rnn2(output, torch.randn(n_layers, len(X_batch), hidden_dim2))
        output, hidden = self.rnn3(output, torch.randn(n_layers, len(X_batch), hidden_dim3))
        return self.linear(output[:,-1])


def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [], [], []
        for X, Y, L in val_loader:                # <-- 3 valores
            preds = model(X, L)                   # <-- pásale lengths
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(
            Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))



def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs+1):
        losses = []
        for X, Y, L in tqdm(train_loader):        # <-- 3 valores
            logits = model(X, L)                  # <-- pásale lengths
            loss = loss_fn(logits, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        CalcValLossAndAccuracy(model, loss_fn, val_loader)


def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    with torch.no_grad():
        for X, Y, L in loader:                    # <-- 3 valores
            preds = model(X, L)                   # <-- pásale lengths
            Y_preds.append(preds)
            Y_shuffled.append(Y)
    gc.collect()
    Y_preds = torch.cat(Y_preds)
    Y_shuffled = torch.cat(Y_shuffled)
    return (Y_shuffled.detach().numpy(),
            F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy())


epochs = 15
learning_rate = 1e-3




# Suponiendo que dataset_df['clase'] contiene las etiquetas originales
class_counts = dataset_df['clase'].value_counts()
target_classes = sorted(class_counts.index.tolist())

# Calcula pesos inversamente proporcionales a la frecuencia
weights = torch.tensor(
    [1.0 / class_counts[cls] for cls in target_classes],
    dtype=torch.float
)

# Normaliza (opcional, pero mantiene magnitud coherente)
weights = weights / weights.sum() * len(weights)

# Define la función de pérdida ponderada
loss_fn = nn.CrossEntropyLoss(weight=weights)


rnn_classifier = RNNClassifier()
class_weights = weights.to(next(rnn_classifier.parameters()).device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(rnn_classifier.parameters(), lr=learning_rate)


TrainModel(rnn_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)



Y_actual, Y_preds = MakePredictions(rnn_classifier, test_loader)

print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_actual, Y_preds, target_names=target_classes))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_actual, Y_preds))

