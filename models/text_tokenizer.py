import json
from functools import lru_cache
from typing import List
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class TextTokenizer:
    """

    """

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        # Mask token behave like a normal word, i.e. include the space before it
        self.mask_token = mask_token

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self._set_special_toke_ids()

    def _set_special_toke_ids(self):
        self.bos_token_id = self._convert_token_to_id(self.bos_token)
        self.eos_token_id = self._convert_token_to_id(self.eos_token)
        self.sep_token_id = self._convert_token_to_id(self.sep_token)
        self.cls_token_id = self._convert_token_to_id(self.cls_token)
        self.unk_token_id = self._convert_token_to_id(self.unk_token)
        self.pad_token_id = self._convert_token_to_id(self.pad_token)
        # Mask token behave like a normal word, i.e. include the space before it
        self.mask_token_id = self._convert_token_to_id(self.mask_token)

        self.all_special_ids = [self.bos_token_id, self.pad_token_id, self.sep_token_id, self.unk_token_id, self.mask_token_id]

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def build_inputs_with_special_tokens_and_mask(
        self, token_ids: List[int]
    ):
        token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        return token_ids

    def convert_ids_to_tokens(
        self, ids, skip_special_tokens
    ):

        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def __call__(self, text, max_length=None):
        tokens = self._tokenize(text)

        token_ids = []
        for tk in tokens:
            token_ids.append(self._convert_token_to_id(tk))

        token_ids = self.build_inputs_with_special_tokens_and_mask(token_ids)
        attention_mask = [1]*len(token_ids)

        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            else:
                num_pad = max_length - len(token_ids)
                token_ids = token_ids + [self.pad_token_id]*num_pad
                attention_mask = attention_mask + [0]*num_pad

        return token_ids, attention_mask
