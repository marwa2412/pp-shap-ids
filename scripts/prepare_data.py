"""Acquire pinned mirror bytes and build the cache expected by the paper code."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_manifest.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path, entry):
    return path.is_file() and path.stat().st_size == entry["size_bytes"] and digest(path) == entry["sha256"]


def acquire(entry, destination, source=None):
    """Never replace an existing source file, including a failed hash check."""
    if destination.exists():
        if not verified(destination, entry):
            raise ValueError(f"Existing source fails SHA-256/size validation: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if source is not None:
            shutil.copyfile(source, temporary)
        else:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "controlled-explanation-disclosure-reproduction"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        if not verified(temporary, entry):
            raise ValueError(f"Downloaded/imported source fails SHA-256/size validation: {entry['path']}")
        if destination.exists():
            raise FileExistsError(destination)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def cache_summary(paths, prefix_rows=160000):
    """Hash ordered logical prefix rows, independently of Arrow record batching."""
    fingerprint = hashlib.sha256()
    total = selected = 0
    schema = None
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            fields = [[field.name, str(field.type)] for field in reader.schema]
            if schema is None:
                schema = fields
                fingerprint.update(json.dumps(fields, separators=(",", ":")).encode("ascii"))
            elif schema != fields:
                raise ValueError("Inconsistent Arrow schemas")
            for batch in reader:
                total += len(batch)
                if selected < prefix_rows:
                    part = batch.slice(0, min(len(batch), prefix_rows-selected))
                    hashes = pd.util.hash_pandas_object(part.to_pandas(), index=False)
                    fingerprint.update(hashes.to_numpy(dtype="<u8").tobytes())
                    selected += len(part)
    return {"rows": total, "prefix_rows": selected, "schema": schema,
            "prefix_logical_sha256": fingerprint.hexdigest()}


def arrow_paths(cache_root, spec):
    folder = cache_root / "datasets" / spec["cache_folder"] / "default/0.0.0" / spec["revision"]
    return folder, [folder / name for name in spec["arrow_names"]]


def write_arrow(sources, spec, staging):
    schema = pa.schema([pa.field(name, pa.type_for_alias(dtype)) for name, dtype in spec["logical_cache"]["schema"]])
    if spec["format"] == "parquet":
        if len(sources) != len(spec["arrow_names"]):
            raise ValueError("Parquet/Arrow shard count mismatch")
        for source, name in zip(sources, spec["arrow_names"]):
            with pa.OSFile(str(staging / name), "wb") as sink, ipc.new_stream(sink, schema) as writer:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=10000):
                    table = pa.Table.from_batches([batch]).cast(schema)
                    writer.write_table(table)
    else:
        name = spec["arrow_names"][0]
        string_fields = {field.name: str for field in schema if pa.types.is_string(field.type)}
        with pa.OSFile(str(staging / name), "wb") as sink, ipc.new_stream(sink, schema) as writer:
            for frame in pd.read_csv(sources[0], dtype=string_fields, chunksize=10000):
                writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
        # The mirror exposes identical train/test aliases, never independent splits.
        shutil.copyfile(staging / name, staging / spec["test_alias"])


def check_arrow(cache_root, spec):
    folder, paths = arrow_paths(cache_root, spec)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Prepared Arrow cache missing: {folder}")
    summary = cache_summary(paths)
    if summary != spec["logical_cache"]:
        raise ValueError(f"Cached rows/schema/order differ from the paper inputs: {folder}")
    if spec.get("test_alias"):
        alias = folder / spec["test_alias"]
        if not alias.is_file() or cache_summary([alias]) != summary:
            raise ValueError("ToN-IoT cached test alias differs from its training table")
    return summary


def prepare(cache_root, source_root=None, check_only=False, manifest=None):
    specs = json.loads(MANIFEST.read_text()) if manifest is None else manifest
    reports = {}
    for name, spec in specs["datasets"].items():
        snapshot = Path("hub") / ("datasets--" + spec["repository"].replace("/", "--")) / "snapshots" / spec["revision"]
        sources = []
        for entry in spec["files"]:
            target = cache_root / snapshot / entry["path"]
            if check_only:
                if not verified(target, entry):
                    raise ValueError(f"Pinned source missing or mismatched: {target}")
            else:
                local = source_root / snapshot / entry["path"] if source_root is not None else None
                acquire(entry, target, local)
            sources.append(target)
        folder, paths = arrow_paths(cache_root, spec)
        if not check_only and not folder.exists():
            folder.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=folder.parent, prefix="preparing-") as temp:
                staging = Path(temp)
                write_arrow(sources, spec, staging)
                if cache_summary([staging / path.name for path in paths]) != spec["logical_cache"]:
                    raise ValueError(f"Conversion differs from recorded logical inputs: {name}")
                staging.rename(folder)
        reports[name] = check_arrow(cache_root, spec)
        print(f"Verified {name}: {reports[name]['rows']:,} cached rows; pinned source hashes and prefix order match", flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline validation only; do not download or create data")
    mode.add_argument("--from-cache", type=Path, help="Import pinned source files from another Hugging Face cache root (offline)")
    args = parser.parse_args()
    try:
        source = args.from_cache.expanduser() if args.from_cache is not None else None
        prepare(Path.home() / ".cache/huggingface", source, args.check)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Data preparation failed: {error}\nNo alternative dataset or revision was substituted.\n")


if __name__ == "__main__":
    main()
