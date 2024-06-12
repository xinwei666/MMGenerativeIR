from SEAL import FMIndex
from transformers import  LlamaTokenizer
import json

corpus = [
    "Doc 1 @@ This is a sample document",
    "Doc 2 @@ This is another sample document",
    "Doc 3 @@ And here you find the final one",
]
labels = ['doc1', 'doc2', 'doc3']

with open("./trie/sent2cid.json", 'r', encoding='utf-8') as f:
    sent2id = json.load(f)

base_model = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(base_model)

def preprocess(doc):
    doc = ' ' + doc
    doc = tokenizer(doc, add_special_tokens=False)['input_ids']
    doc += [tokenizer.eos_token_id]
    return doc

corpus_tokenized = [preprocess(doc) for doc in corpus]

index = FMIndex()
index.initialize(corpus_tokenized, in_memory=True)
index.labels = labels

index.save('res/sample/sample_corpus.fm_index')
# writes res/sample/sample_corpus.fm_index.fmi
# writes res/sample/sample_corpus.fm_index.oth

index = FMIndex.load('res/sample/sample_corpus.fm_index')