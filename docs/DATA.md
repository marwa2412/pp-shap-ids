# Dataset Identity

The code uses the exact pinned mirrors recorded in the paper, not whichever
UNSW-NB15 or ToN-IoT download happens to be available:

| Dataset | Mirror | Revision |
| --- | --- | --- |
| UNSW-NB15 | [Mouwiya/UNSW-NB15](https://huggingface.co/datasets/Mouwiya/UNSW-NB15/tree/8ad6b19373f143086a66f52b0d6035a6959d83a9) | `8ad6b19373f143086a66f52b0d6035a6959d83a9` |
| ToN-IoT network | [codymlewis/TON_IoT_network](https://huggingface.co/datasets/codymlewis/TON_IoT_network/tree/b953d9388d2db3bc392ea7150476e4f219a2a1cb) | `b953d9388d2db3bc392ea7150476e4f219a2a1cb` |

`data_manifest.json` pins two Parquet shards and one CSV, approximately 260 MB
in total. Derived Arrow caches require additional disk space. The helper
requires read access to these files; a remote availability or permission
failure is reported rather than bypassed. No dataset content is in the Git
release. This package grants no software license or dataset rights. The
datasets remain subject to their own terms, regardless of code licensing.

The UNSW mirror has 2,280,090 rows, not the original full corpus's 2,540,044.
The ToN CSV has 211,043 rows and no timestamp field. Its original cached
train/test aliases are identical, so preparation creates identical aliases
and experiments create their own audited splits. Neither alias is an
independent validation set. Original-release byte equivalence is unverified.

## Byte Identity Versus Logical Identity

Raw Parquet/CSV files must match their original SHA-256 and byte counts.
Arrow files are rebuilt with PyArrow and can differ from the original
Hugging Face cache serialization in metadata or record-batch boundaries.
Those byte differences are not hidden or claimed to be identical artifacts.

The helper compares ordered field names/types, total rows, and the logical
row fingerprint of the first 160,000 records against the retained cache.
The fingerprint uses pinned pandas 3.0.2 row hashing and includes values and
order, but excludes the DataFrame index. The experiments assign the same
positional row IDs. The first 160,000 rows are the complete input pool for
the initial experiment and source-record replay. The larger structured
holdouts read the byte-verified raw Parquet/CSV sources directly.

New experiments record new Arrow file hashes. Replaying an old ledger that
references a different Arrow byte serialization requires that ledger's
original retained source files; matching logical prefixes is not permission
to rewrite its provenance hashes.

## Storage and Configuration

The historical scripts use the current user's `~/.cache/huggingface` layout.
The helper prepares that same layout without modifying the algorithms.
Use an isolated user/container home for a completely separate reproduction
cache. The helper will not overwrite an existing incomplete or mismatched
revision directory; inspect that directory or use a fresh isolated home.
Do not substitute a random public mirror or a differently ordered export.
