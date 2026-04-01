import torch
from torch.utils.data import Dataset
from collections import Counter

class TextSequenceDataset(Dataset):
    def __init__(self, texts, seq_length=5, min_freq=2, existing_vocab=None):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []
        
        if existing_vocab is None:
            # Build vocab
            words = []
            for text in texts:
                words.extend(text.lower().split())
            
            freqs = Counter(words)
            self.vocab = {"<pad>": 0, "<unk>": 1}
            idx = 2
            for w, c in freqs.items():
                if c >= min_freq:
                    self.vocab[w] = idx
                    idx += 1
        else:
            self.vocab = existing_vocab
                
        self.idx2word = {i: w for w, i in self.vocab.items()}
        
        # Create sequences
        for text in texts:
            tokens = [self.vocab.get(w, self.vocab["<unk>"]) for w in text.lower().split()]
            if len(tokens) <= seq_length:
                continue
            for i in range(len(tokens) - seq_length):
                self.inputs.append(tokens[i:i+seq_length])
                self.targets.append(tokens[i+seq_length])
                
        if self.inputs:
            self.inputs = torch.tensor(self.inputs, dtype=torch.long)
            self.targets = torch.tensor(self.targets, dtype=torch.long)
        else:
            self.inputs = torch.empty((0, seq_length), dtype=torch.long)
            self.targets = torch.empty((0,), dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
