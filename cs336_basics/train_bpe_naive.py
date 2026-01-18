import regex as re
import json
from collections import Counter, defaultdict
from itertools import islice

# GPT-2 style regex for pre-tokenization
_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"        


def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    训练字节级 BPE tokenizer
    返回 vocab (id->bytes) 和 merges ([(b1,b2), ...])
    """
    # 初始化 vocab: 所有 byte 值
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    # 添加特殊 tokens
    for tok in special_tokens:
        vocab[next_id] = tok.encode('utf-8')
        next_id += 1

    # 预分词并统计 pre-token 出现次数
    pattern = re.compile(_PAT)
    doc_delim = '|'.join(map(re.escape, special_tokens)) if special_tokens else None
    freq: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [text] if not doc_delim else re.split(doc_delim, text)
    for chunk in chunks:
        for m in pattern.finditer(chunk):
            token = m.group(0)
            bseq = tuple(bytes([b]) for b in token.encode("utf-8"))
            freq[bseq] += 1

    merges: list[tuple[bytes, bytes]] = []
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for bseq, count in freq.items():
        for a, b in zip(bseq, bseq[1:]):
            pair_counts[(a,b)] += count
    # 计算合并直到达到 vocab_size
    while next_id < vocab_size:
        # 统计所有相邻字节对频率
        if not pair_counts:
            break
        # 找到频率最高，按字节对比序列大者优先
        max_freq = max(pair_counts.values())
        candidates = [pair for pair, c in pair_counts.items() if c == max_freq]
        merge_pair = max(candidates)
        merges.append(merge_pair)
        pair_counts.pop(merge_pair)
        
        A, B = merge_pair
        new_word = A + B

        for bseq, cnt in freq.copy().items():
            bseq_len = len(bseq)
            new_seq = []
            changed = False
            i = 0
            while i < bseq_len:
                if i < bseq_len - 1 and bseq[i] == A and bseq[i+1] == B:
                    new_seq.append(new_word)
                    changed = True
                    if i > 0:
                        pair_counts[(bseq[i-1], A)] -= cnt
                        pair_counts[(bseq[i-1], new_word)] += cnt
                    if i < bseq_len - 2:
                        pair_counts[(B, bseq[i+2])] -= cnt
                        pair_counts[(new_word, bseq[i+2])] += cnt
                    i += 2
                else:
                    new_seq.append(bseq[i])
                    i += 1
            if changed:
                freq.pop(bseq)
                freq[tuple(new_seq)] += cnt
        # 将新 token 加入 vocab
        vocab[next_id] = new_word
        next_id += 1
    return vocab, merges

def train_bpe(datafile_name, vocab_size):
    data_dir = None # # NOTE: set your data dir path
    input_path = f'{data_dir}{datafile_name}.txt'
    vocab, _ = train_bpe_tokenizer(input_path, vocab_size, ["<|endoftext|>"])
    vocab_path = f'./{datafile_name}-{vocab_size}.txt'
    max_len = 0
    max_token = ''
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for idx, token_bytes in vocab.items():
            # 将 bytes 转为可读字符串（如 b'hello' -> 'hello'）
            try:
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                token_str = str(token_bytes)  # 非UTF-8字节回退到原始表示
            f.write(f"{idx}\t{token_str}\n")
            if max_len < len(token_str):
                max_token = token_str
                max_len = len(token_str)
    return max_token

import time

if __name__ == "__main__":
    
    st = time.time()
    token  = train_bpe('TinyStoriesV2-GPT4-train', 10000)
    end = time.time()
    print("train TinyStoriesV2  time consuming: ", (end - st) / 60, "min")
    print(f"the longest token: {token}")

    
    st = time.time()
    token  = train_bpe('owt_train', 32000)
    end = time.time()
    print("train OpenWebText  time consuming: ", (end - st) / 60, "min")
    print(f"the longest token: {token}")