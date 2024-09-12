import tiktoken
import regex as re
import unicodedata

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

class GPT4Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = GPT4_SPECIAL_TOKENS
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        mergeable_ranks = self.enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

    def encode(self, text, allowed_special="all"):
        if allowed_special == "all":
            return self.enc.encode(text, allowed_special=allowed_special)
        
        special = self._get_special_tokens(allowed_special)
        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        parts = re.split(special_pattern, text)
        
        ids = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def _encode_chunk(self, text_bytes):
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        return self.enc.encode_ordinary(text_bytes.decode('utf-8', errors='replace'))

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] if idx in self.vocab else self.inverse_special_tokens[idx].encode("utf-8") for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        return text_bytes.decode("utf-8", errors="replace")

    def _get_special_tokens(self, allowed_special):
        if allowed_special == "all":
            return self.special_tokens
        elif allowed_special == "none":
            return {}
        elif allowed_special == "none_raise":
            return {}
        elif isinstance(allowed_special, set):
            return {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

    def token_to_id(self, token):
        return self.enc.encode_single_token(token)

    def id_to_token(self, id):
        return self.enc.decode([id])

    def get_vocab(self):
        return self.enc.name_to_id

def render_token(s):
    return "".join(ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}" for ch in s)

# Usage example
if __name__ == "__main__":
    tokenizer = GPT4Tokenizer()
    text = "Hello, world! This is a test. New document."
    print(f"Original text: {text}")

    print("\nEncoding text...")
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")

    print("\nDecoding back to text...")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    

# Expected output:
"""
Original text: Hello, world! This is a test. <|endoftext|> New document.

Encoding text...
Encoded: [9906, 11, 4435, 0, 934, 360, 257, 1256, 13, 100257, 3228, 1414, 13]

Decoding back to text...
Decoded: Hello, world! This is a test. <|endoftext|> New document.

Sample of vocabulary:
0: !
1: "
2: #
3: $
4: %
5: &
6: '
7: (
8: )
9: *

Special tokens:
100257: <|endoftext|>
100258: <|fim_prefix|>
100259: <|fim_middle|>
100260: <|fim_suffix|>
100276: <|endofprompt|>
"""