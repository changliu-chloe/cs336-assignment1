import regex as re
from collections import defaultdict
from typing import BinaryIO, Self
from functools import reduce
import os
import heapq
import multiprocessing as mp
import pickle
import time

# GPT-2 style regex for pre-tokenization
PAT = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


class ReverseLexOrderPair:
    """
    Encapsulates (bytes, bytes) so that in a min-heap, the "largest in normal lex order"
    is treated as the smallest. Ensures that tie frequencies pop in reverse lex order.
    """

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: Self) -> bool:
        # Invert normal order: self < other if self is > other (so larger lex sorts first).
        return self.pair > other.pair

    def __eq__(self, other: Self) -> bool:
        return self.pair == other.pair
    

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(chunk: str, special_pattern: re.Pattern | None) -> dict[tuple[bytes], int]:
    """Regex tokenizes the chunk. Splits first on special tokens, then uses PAT."""
    freqs: dict[tuple[bytes], int] = defaultdict(int)
    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk]

    for sub_chunk in sub_chunks:
        for word in PAT.finditer(sub_chunk):
            word_bytes = tuple(bytes([b]) for b in word.group().encode("UTF-8"))
            freqs[word_bytes] += 1
    return freqs

def merge_freq_dicts(dict1: dict[tuple[bytes], int], dict2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    "Adds frequences from dict2 to dict1"
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result


def pre_tokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """
    Splits a file into chunks aligned with <|endoftext|>, then tokenizes each chunk
    in parallel. Returns aggregated frequency dict.
    """
    num_processes = 24
    # num_processes = mp.cpu_count()
    print(f"cpu count: {num_processes}")
    pool = mp.Pool(processes=num_processes)
    chunk_freqs = []
    special_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens)) if special_tokens else None

    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Read each chunk in bytes, decode, then apply_async for parallel tokenization
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on each chunk and store the counts for each pre-token
            chunk_freqs.append(pool.apply_async(pre_tokenize_chunk, (chunk, special_pattern)))
    pool.close()
    pool.join()

    freq_dicts = [res.get() for res in chunk_freqs]
    combined_freqs = reduce(merge_freq_dicts, freq_dicts, {})
    return combined_freqs


def get_pair_freqs(
        freqs: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """
    Builds a pair-frequency table and reverse mapping (pair -> set of keys).
    """
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set] = defaultdict(set)

    for key_word, freq in freqs.items():
        for i in range(len(key_word) - 1):
            pair = (key_word[i], key_word[i+1])
            pair_freqs[pair] += freq
            pairs_to_keys[pair].add(key_word) 
    return pair_freqs, pairs_to_keys


def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    new_repr = []
    item = pair[0] + pair[1]
    i = 0
    n = len(old_repr)
    while i < n:
        if i < n - 1 and old_repr[i: i+2] == pair:
            new_repr.append(item)
            i += 2
        else:
            new_repr.append(old_repr[i])
            i += 1
    return tuple(new_repr)


def update_freqs(
        freqs: dict[tuple[bytes], int],
        pair_freqs: dict[tuple[bytes, bytes], int],
        pairs_to_keys: dict[tuple[bytes, bytes], set],
        pair: tuple[bytes, bytes],
) -> set[tuple[bytes, bytes]]:
    new_pairs = set()
    keys_to_mod = pairs_to_keys[pair].copy()
    merged_token = pair[0] + pair[1]

    for old_key in keys_to_mod:
        old_freq = freqs.pop(old_key)
        new_key = build_new_repr(old_key, pair)
        freqs[new_key] = freqs.get(new_key, 0) + old_freq

        # modify pair_freq 
        for i in range(len(old_key) - 1):
            cur_pair = old_key[i:i+2]
            pair_freqs[cur_pair] -= old_freq
            if pair_freqs[cur_pair] <= 0:
                del pair_freqs[cur_pair]
            pairs_to_keys[cur_pair].discard(old_key)
        for i in range(len(new_key) - 1):
            cur_pair = new_key[i:i+2]
            pair_freqs[cur_pair] += old_freq
            if merged_token in cur_pair:
                new_pairs.add(cur_pair)
            pairs_to_keys[cur_pair].add(new_key)
    pairs_to_keys[pair] = set() # remove pair_keys
    return new_pairs


def write_merges(merges, outpath):
    """Pickle the merges list to a binary file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved {len(merges)} merges to {outpath}")


def write_vocab(vocab, outpath):
    """Pickle the vocab dict to a binary file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary with {len(vocab)} tokens to {outpath}")


def train_bpe(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str],
        merges_outpath: str = None,
        vocab_outpath: str = None,
):
    """
    训练字节级 BPE tokenizer
    返回 vocab (id->bytes) 和 merges ([(b1,b2), ...])
    """
    
    train_start_time = time.time()
    # 初始化 vocab: 所有 byte 值
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    # 添加特殊 tokens
    for tok in special_tokens:
        vocab[next_id] = tok.encode('utf-8')
        next_id += 1

    # 预分词并统计 pre-token 出现次数
    print("Pretokenize start")
    start_time = time.time()
    freqs = pre_tokenize(input_path, special_tokens)
    print(f"Pretokenize finished in {time.time() - start_time:.2f}s")

    print("Initial pair frequences start")
    start_time = time.time()
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)
    # build a max_heap
    pair_heap = []
    for p, f in pair_freqs.items():
        if f > 0:
            heapq.heappush(pair_heap, (-f, ReverseLexOrderPair(p), p))
    print(f"Initial pair frequencies finished in {time.time() - start_time:.2f}s")    

    print("Merge start")
    start_time = time.time()
    # 计算合并直到达到 vocab_size
    merges: list[tuple[bytes, bytes]] = []
    while next_id < vocab_size:
        # 统计所有相邻字节对频率
        if not pair_freqs:
            break
        # 找到频率最高，按字节对比序列大者优先
        while pair_heap:    # 循环查找，因为首个弹出的pair可能不符合要求 when?
            neg_freq, _, top_pair = heapq.heappop(pair_heap)
            freq = -neg_freq
            if pair_freqs[top_pair] == freq:    # when not equal?
                pair = top_pair
                break
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[top_pair], ReverseLexOrderPair(top_pair), top_pair))
        else:
            break
        if pair_freqs[pair] <= 0:
            break

        A, B = pair
        new_word = A + B
        merges.append(pair)
        vocab[next_id] = new_word
        next_id += 1
        
        # merge and update freq table
        # 从pairs_to_keys中拿到所有需要修改的‘旧词汇’，对于每一个‘旧词汇’：
        #   1. 从freqs中拿到‘旧词汇’的频率，并用‘新词汇new_repr’替代‘旧词汇’
        #   2. 从pairs_to_keys的keys中删除‘旧词汇’
        #   3. 从‘新词汇’中拿到三种新的token_pair：(A, pair), (pair, B), (pair, pair)
        #   4. 从pair_freqs中更新这些new_pair的频率，保证pair_freqs里的频率一定是正确的，但是pair_heap里会有未更新的脏数据
        
        new_pairs = update_freqs(freqs, pair_freqs, pairs_to_keys, pair)
        for cp in new_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[cp], ReverseLexOrderPair(cp), cp))

    print(f"Merges completed in {time.time() - start_time:.2f}s")
    print(f"Training completed in {time.time() - train_start_time:.2f}s")
    
    # Optionally save merges and vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return vocab, merges


def train_bpe_(datafile_name, vocab_size, data_file_path='./'):
    input_path = data_file_path + datafile_name + '.txt'
    output_vocab_path = f'./out/{datafile_name}-vocab-{vocab_size}.txt'
    output_merge_path = f'./out/{datafile_name}-merges.txt'
    _, _ = train_bpe(input_path, vocab_size, ["<|endoftext|>"], output_merge_path, output_vocab_path)


if __name__ == "__main__":
    data_file_path = None   # NOTE: set your data dir path

    train_bpe_('TinyStoriesV2-GPT4-train', 10000, data_file_path)
    train_bpe_('owt_train', 32000, data_file_path)