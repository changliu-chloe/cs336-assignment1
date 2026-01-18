from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics.train_bpe import PAT

class Tokenizer:
    def __init__(
           self, 
           vocab: dict[int, bytes], 
           merges: list[tuple[bytes, bytes]], 
           special_tokens: list[str] | None = None
    ):
        """
        Constructs a tokenizer from a vocab, list of merges, and (optionally) list of special tokens.
        """
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        self.encode_cache = {}
        self.cache_hits = 0

        self.pretokenize_pattern = re.compile(PAT)

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"

            next_id = len(self.vocab)
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1
        else:
            self.special_pattern = None
            self.special_tokens = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    

    def encode(self, text: str) -> list[int]:

        if not self.special_tokens:
            return self._encode_chunk(text)
        
        special_chunks = re.split(self.special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                ids.append(self.vocab_inv[part.encode("UTF-8")])
            else:
                ids.extend(self._encode_chunk(part))
        return ids

    
    def _encode_chunk(self, text: str) -> list[int]:
        pretokens = self._pretokenize(text)
        pretoken_reprs: dict[str, tuple[bytes]] = {}

        ids = []

        for p in pretokens:
            if p in self.encode_cache:
                ids.extend(self.encode_cache[p])
                self.cache_hits += 1
            else:
                if p not in pretoken_reprs:
                    match_bytes = tuple(bytes([b]) for b in p.encode("UTF-8"))
                    pretoken_reprs[p] = match_bytes
                
                merged = self._merge_subword(pretoken_reprs[p])
                token_ids = [self.vocab_inv[token] for token in merged]
                self.encode_cache[p] = token_ids
                ids.extend(token_ids)
        return ids


    def _merge_subword(self, rep: tuple[bytes]) -> tuple[bytes]:
        """
        Given a list of subword units (bytes), repeatedly merges adjacent pairs
        in ascending rank order until no more merges are found.
        """
        # why in ascending rank ?  because lower rank means higher priority
        while True:
            best_rank = float("inf")
            best_idx = None

            for i in range(len(rep) - 1):
                pair = rep[i:i+2]
                rank = self.merges_dict.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx is None: 
                break
            merged = (rep[best_idx] + rep[best_idx + 1],)   # 在 Python 中，tuple(bytes_object) 会将字节串拆解为整数元组
            rep = rep[:best_idx] + merged + rep[best_idx+2:]
        return rep

    
    def _pretokenize(self, text: str) -> list[str]:
        pretokens: list[str] = []

        for match in self.pretokenize_pattern.finditer(text):
            pretokens.append(match.group())
        
        return pretokens
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Yields token IDs lazily from an iterable of strings (e.g., a file handle)."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")
    