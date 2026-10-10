"""Audit-log integrity relative to an independently retained checkpoint."""

import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def genesis(log_id):
    return digest({"format": "artiis-audit-v1", "log_id": log_id})


def append(entries, payload, log_id):
    previous = entries[-1]["hash"] if entries else genesis(log_id)
    body = {"log_id": log_id, "sequence": len(entries),
            "previous_hash": previous, "payload": payload}
    # Round-trip detaches mutable caller objects from committed records.
    record = json.loads(canonical({**body, "hash": digest(body)}))
    entries.append(record)
    return record


def checkpoint(entries, log_id):
    return {"format": "artiis-checkpoint-v1", "log_id": log_id,
            "count": len(entries),
            "head": entries[-1]["hash"] if entries else genesis(log_id)}


def verify(entries, trusted):
    """Verify an exact snapshot; the caller must authenticate the checkpoint."""
    try:
        if trusted["format"] != "artiis-checkpoint-v1":
            return False, "checkpoint_format"
        if len(entries) != trusted["count"]:
            return False, "record_count"
        previous = genesis(trusted["log_id"])
        for i, entry in enumerate(entries):
            if set(entry) != {"log_id", "sequence", "previous_hash", "payload", "hash"}:
                return False, f"record_shape:{i}"
            if entry["log_id"] != trusted["log_id"] or entry["sequence"] != i:
                return False, f"identity_or_sequence:{i}"
            if entry["previous_hash"] != previous:
                return False, f"chain_link:{i}"
            body = {key: value for key, value in entry.items() if key != "hash"}
            if digest(body) != entry["hash"]:
                return False, f"record_hash:{i}"
            previous = entry["hash"]
        if previous != trusted["head"]:
            return False, "trusted_head"
        return True, "ok"
    except (KeyError, TypeError, ValueError):
        return False, "invalid_record"


def verify_artifacts(payload, root):
    for name, item in payload["artifacts"].items():
        path = root / item["path"]
        if not path.is_file() or file_digest(path) != item["sha256"]:
            return False, name
    return True, "ok"
