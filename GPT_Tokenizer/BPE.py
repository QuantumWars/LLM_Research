#Byte-Pair Encoding
text = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide.
We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). 
But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, 
and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, 
even 30 years after Unicode’s inception."""

tokens = text.encode("utf-8")
tokens = list(map(int,tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts


#in the list of ints(ids), replace all consecutive occurences of pair with the new token idx
def merge(ids,pair,idx):
    newids = []
    i = 0 
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids



vocal_size = 276 #the desired final vocabulary size
num_merges = vocal_size-256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats,key=stats.get)
    idx = 256+i
    print(f"Merging {pair} into a new token {idx} because the value is {stats.get(pair)}")
    ids = merge(ids,pair,idx)
    merges[pair] = idx

# print("token lenght",len(tokens))
# print("ids length",len(ids))
# print(f"coimpression ratio : {len(tokens) / len(ids):.2f}X")


vocab = {idx:bytes([idx]) for idx in range(256)}
for (p0,p1),idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8",errors="replace")
    return text

def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

valtext = """Many common characters, including numerals, punctuation, and other symbols, 
are unified within the standard and are not treated as specific to any given writing system. 
Unicode encodes thousands of emoji, with the continued development thereof conducted by the 
Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in 
large part responsible for the initial popularization of emoji outside of Japan. Unicode is
   ultimately capable of encoding more than 1.1 million characters."""
valtext2 = decode(encode(valtext))
# print(valtext2 == valtext)

import regex as re

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
print(re.findall(gpt2pat,"how is your dog?"))