import json
import os
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise

import regex as re
from tqdm import tqdm
from pathlib import Path


class BPE:
    def __init__(self, vocab_size: int):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for base byte vocabulary.")

        self.vocab_size = vocab_size

        # Maps (token_id1, token_id2) -> new_token_id
        self.merges: dict[tuple[int, int], int] = {}

        # Maps token_id -> bytes
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        # Special Tokens
        self.special_tokens = {"<BOS>": 256,"<EOS>": 257, "<PAD>": 258}
        special_escaped = "|".join(re.escape(tok) for tok in self.special_tokens.keys())
        self.special_pattern = re.compile(f"({special_escaped})")
        self.vocab.update({v: k.encode("utf-8") for k, v in self.special_tokens.items()})

        # Cache for fast encode inference
        self._encode_cache: dict[str, list[int]] = {}

        # Pre-tokenization regex
        self.pattern = re.compile(
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
            r"""|[^\r\n\p{L}\p{N}]?+\p{L}+"""
            r"""|\p{N}{1,3}"""
            r"""| ?[^\s\p{L}\p{N}]++[\r\n]*"""
            r"""|\s*[\r\n]+"""
            r"""|\s+(?!\S)"""
            r"""|\s+"""
        )

    @staticmethod
    def _merge_tuple(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        """Fast tuple replacement of consecutive pair matches."""
        new_ids = []
        i = 0
        n = len(ids)
        p0, p1 = pair
        while i < n:
            if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return tuple(new_ids)

    def _save(self, save_dir: str,prefix: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        save_dir = Path(save_dir)
        # Vocab Save
        serializable_vocab = {
            token_id: byte_val.hex() for token_id, byte_val in self.vocab.items()
        }
        vocab_path = save_dir / f"{prefix}.vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, indent=2)

        # Merges Save
        serializable_merges = {
            f"{p0} {p1}": new_id
            for (p0, p1), new_id in sorted(self.merges.items(), key=lambda item: item[1])
        }
        merges_path = save_dir / f"{prefix}.merges.json"
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f, indent=2)

        # Vocab Readable Save
        readable_vocab = {
            token_id: byte_val.decode("utf-8",errors="replace") for token_id, byte_val in self.vocab.items()
        }
        vocab_readable_path = save_dir / f"{prefix}.vocab_readable.json"

        with open(vocab_readable_path, "w", encoding="utf-8") as f:
            json.dump(readable_vocab, f, indent=2)
        print(f"Saved: {vocab_path} and {merges_path} and {vocab_readable_path}")

    @classmethod
    def load(cls, load_dir : str,prefix: str):
        """Loads merges and vocab from disk and constructs a BPE instance."""
        load_dir = Path(load_dir)

        vocab_path = load_dir / f"{prefix}.vocab.json"
        merges_path = load_dir / f"{prefix}.merges.json"

        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges = {}
        for key_pair, new_id in raw_merges.items():
            p0_str, p1_str = key_pair.split()
            merges[(int(p0_str), int(p1_str))] = int(new_id)

        tokenizer = cls(vocab_size=len(vocab))
        tokenizer.merges = merges
        tokenizer.vocab = vocab
        return tokenizer

    def train(self,
              texts: Iterable[str],
              save_dir:str,
              prefix: str,
              min_frequency: int = 2,
              verbose: bool = False):
        """
        Fast & memory-efficient training:
        Aggregates identical chunks into weighted frequencies instead of storing redundant lists.
        """
        word_counts = Counter()
        print("Pre-tokenizing texts into frequency dictionary...")
        for text in texts:
            # Count string tokens directly to prevent duplicating byte lists in RAM
            word_counts.update(self.pattern.findall(text))

        # Convert unique strings to tuple of raw byte ints with their corpus count
        vocab_words: dict[tuple[int, ...], int] = {
            tuple(word.encode("utf-8")): count for word, count in word_counts.items()
        }
        del word_counts  # Free memory immediately

        num_merges = self.vocab_size - len(self.vocab)
        last_token_idx = len(self.vocab)

        print(f"Unique subwords to train on: {len(vocab_words)}")
        print(f"Starting BPE training: {num_merges} merges planned.")

        pbar = tqdm(range(num_merges), desc="Learning BPE Merges", unit="merge")
        for i in pbar:
            # 1. Count pair occurrences weighted by word frequency
            stats = Counter()
            for word, freq in vocab_words.items():
                for pair in pairwise(word):
                    stats[pair] += freq

            if not stats:
                pbar.write(f"No more pairs to merge. Stopping early at merge {i}.")
                break

            best_pair,best_freq = stats.most_common(1)[0]
            if best_freq < min_frequency:
                pbar.write(f"Most frequent pair {best_pair} has frequency {best_freq} < {min_frequency}. Stopping early at merge {i}.")
                break
            new_id = last_token_idx + i

            # 2. Record merge & vocabulary
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 3. Update only words that contain the target pair
            p0, p1 = best_pair
            new_vocab_words = {}
            for word, freq in vocab_words.items():
                # Fast presence check before running substitution
                has_pair = any(word[idx] == p0 and word[idx + 1] == p1 for idx in range(len(word) - 1))
                if has_pair:
                    new_word = self._merge_tuple(word, best_pair, new_id)
                    new_vocab_words[new_word] = freq
                else:
                    new_vocab_words[word] = freq

            vocab_words = new_vocab_words
            pbar.set_postfix({"Vocab Size": len(self.vocab)})

            if verbose and (i + 1) % 50 == 0:
                pbar.write(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id}")

        pbar.close()
        print("Training complete.")
        self._save(save_dir,prefix)

    def _encode_chunk(self, chunk_str: str) -> list[int]:
        """Encodes a single pre-tokenized regex chunk using learned merges with caching."""
        if chunk_str in self._encode_cache:
            return self._encode_cache[chunk_str]

        chunk_ids = list(chunk_str.encode("utf-8"))
        while len(chunk_ids) >= 2:
            stats = pairwise(chunk_ids)
            pair_candidates = {pair: self.merges[pair] for pair in stats if pair in self.merges}
            if not pair_candidates:
                break
            best_pair = min(pair_candidates, key=lambda p: pair_candidates[p])
            chunk_ids = list(self._merge_tuple(tuple(chunk_ids), best_pair, self.merges[best_pair]))

        # Cache top repetitive subwords (cap cache at 50,000 entries)
        if len(self._encode_cache) < 50000:
            self._encode_cache[chunk_str] = chunk_ids

        return chunk_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenizes an input string into token IDs using learned merges."""
        if not text:
            return [self.special_tokens["<EOS>"]] if add_special_tokens else []

        parts = self.special_pattern.split(text)
        encoded_ids = []

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                encoded_ids.append(self.special_tokens[part])
            else:
                for chunk in self.pattern.findall(part):
                    encoded_ids.extend(self._encode_chunk(chunk))

        if add_special_tokens:
            eos_id = self.special_tokens["<EOS>"]
            if not encoded_ids or encoded_ids[-1] != eos_id:
                encoded_ids.append(eos_id)

        return encoded_ids

    def decode(self, ids: list[int], keep_special_tokens: bool = False) -> str:
        """Decodes token IDs back into a UTF-8 string."""
        if keep_special_tokens:
            raw_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        else:
            raw_bytes = b"".join(
                self.vocab[idx] for idx in ids
                if idx in self.vocab and idx not in self.special_tokens.values()
            )
        return raw_bytes.decode("utf-8", errors="replace")