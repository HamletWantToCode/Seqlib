import unicodedata
import string
import re
import torch
from torch.utils.data import Dataset


__all__ = ['LangDataset', 'SOS_token', 'EOS_token']

SOS_token = 0
EOS_token = 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.max_len_sentence = 0

    def addSentence(self, sentence):
        if len(sentence) > self.max_len_sentence:
            self.max_len_sentence = len(sentence)
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# transformer
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    for _ in range(lang.max_len_sentence - len(sentence)):
        indexes.append(0)
    return torch.tensor(indexes, dtype=torch.long)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# pre-filter
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0]) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH and \
        ' '.join(p[0]).startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


class LangDataset(Dataset):
    def __init__(self, root_dir, lang1=None, lang2=None, reverse=False):

        init_dict = {
            'lang1': lang1,
            'lang2': lang2,
            'reverse': reverse
        }

        self.input_lang, self.output_lang, self.data = self.loader(root_dir, **init_dict)


    def loader(self, root_dir, lang1=None, lang2=None, reverse=False):
        with open(root_dir+'/%s-%s.txt' %(lang1, lang2), encoding='utf-8') as f:
            _data = f.read().strip().split('\n')
        
        _pairs = [[normalizeString(s) for s in l.split('\t')] for l in _data]
        _pairs = [[s.split(' ') for s in l] for l in _pairs]
        pairs = filterPairs(_pairs)
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        return input_lang, output_lang, pairs
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :feature, target: seq_len
        """
        _feature, _target = self.data[idx]
        pair = (_feature, _target)

        feature, target = tensorsFromPair(pair, self.input_lang, self.output_lang)
        return (feature, target)

