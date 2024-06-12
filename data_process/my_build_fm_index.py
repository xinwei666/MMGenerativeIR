from seal import FMIndex
from transformers import LlamaTokenizer
import json
import csv

def load_csv(path):
    corpus_dict = {}
    with open(path, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',',)
        for row in spamreader:
            sent = row[1].replace('\n', ' ').strip()
            if sent == 'text':
                continue
            # sent_list = sentence_token_nltk(sent)
            #corpus_dict[row[0]] = sent_list
            corpus_dict[row[0]] = sent

    return corpus_dict

corpus = load_csv('./dataset/okvqa/okvqa_train_corpus.csv')

sent_ids = {}
for cid in corpus:
    sent = corpus[cid]
    sent_ids[sent] = cid

corpus = []
labels = []

for sent in sent_ids:
    cid = sent_ids[sent]
    sent = sent.replace('\n', ' ')
    sent = sent.strip()
    corpus.append(sent)
    labels.append(str(cid))

id2text = {sent_ids[key]: key for key in sent_ids}


with open("./index/id2text.json", 'w', encoding='utf-8') as f:
    json.dump(id2text, f)

base_model = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(base_model)

def preprocess(doc):
    doc = [1] + tokenizer.encode(doc)[1:] + [2]
    return doc

corpus_tokenized = [preprocess(doc) for doc in corpus]

index = FMIndex()
index.initialize(corpus_tokenized, in_memory=True)
index.labels = labels

index.save('./index/okvqa_corpus12k.fm_index')

# index = FMIndex.load('./index/sample_corpus.fm_index')