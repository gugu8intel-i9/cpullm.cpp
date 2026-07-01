# Tokenizer Parity

cpullm.cpp now has a lightweight GGUF-aware tokenizer core designed for high runtime efficiency and low memory.

## Implemented

- GGUF `tokenizer.ggml.tokens` loading
- GGUF `tokenizer.ggml.merges` loading when present
- cached token-to-id map
- compact longest-match trie fallback
- byte fallback for unknown text spans
- byte-level BPE merge loop when merge ranks are available
- tokenizer status reporting

## Fast paths

1. **Merge BPE path**: uses cached merge ranks and token IDs.
2. **Trie path**: longest-match tokenization without repeatedly hashing substrings.
3. **Byte fallback**: keeps unsupported bytes representable without large tables.

## No mock policy

If GGUF merge ranks are missing, cpullm does not claim exact BPE parity. The production planner reports:

```text
tokenizer.merges_missing
```

The tokenizer still works using the longest-match trie and byte fallback, but exact parity requires the model's merge data.

## Why this is lightweight

The tokenizer stores:

- the token strings already present in GGUF metadata
- a token ID hash map
- a compact trie of token bytes
- merge-rank pairs only when present

It avoids heavyweight external tokenizer dependencies.
