# Binary-Compressor-Decompressor

A Python-based binary compression and decompression tool that reduces repeated and near-matching 32-bit instruction patterns using multiple encoding strategies.

## Project Overview

This project implements a custom compression format for binary instruction data. It combines dictionary-based encoding, mismatch-aware transformations, and run-length encoding to produce compact output, then reconstructs the original stream through a matching decompression flow.

## Technical Highlights

- Designed a tagged binary encoding format with dedicated code paths per compression strategy
- Implemented dictionary construction and dictionary-index mapping for frequent binary patterns
- Built compression/decompression structures to keep parsing and transformation logic organized
- Implemented reversible decompression logic for each encoded tag type
- 
## Compression Techniques

- `Direct Match`: references dictionary entries by index
- `One-Bit Mismatch`: encodes one flipped bit relative to a dictionary entry
- `Two-Bit Consecutive Mismatch`: encodes two adjacent flipped bits
- `Four-Bit Consecutive Mismatch`: encodes a 4-bit flipped window
- `Two-Bit Anywhere Mismatch`: encodes two flipped bits at arbitrary positions
- `Bitmask Encoding`: encodes mismatch patterns within a 4-bit window
- `Run-Length Encoding (RLE)`: compresses repeated consecutive lines
- `Original Fallback`: stores uncompressed 32-bit lines when no method is better

## Core Files

- `SIM.py`: main compression/decompression logic, tag parsing, dictionary mapping, and file output flow
- `original.txt`: source binary input stream
- `compressed.txt`: encoded reference/compressed data input format
- `cout.txt`: generated compression output
- `dout.txt`: generated decompression output
- `reference.txt`: annotated reference examples and dictionary mapping notes

## Lessons Learned

- **Binary format design:** building an encoding scheme that balances compactness and decoding clarity
- **Bit-level data handling:** representing mismatches and transformations through precise bit operations
- **Dictionary-driven optimization:** leveraging frequent pattern reuse for higher compression efficiency
- **Algorithm tradeoff thinking:** selecting between multiple encoding strategies based on pattern characteristics
