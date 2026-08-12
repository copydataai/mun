from __future__ import annotations

import base64
from hashlib import sha256
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_private_key

from .artifacts import canonical_json_bytes
from .errors import MunError


class AuthenticationError(MunError):
    pass


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def sign_artifact(
    artifact: Mapping[str, Any],
    receipt: Mapping[str, Any],
    private_key_pem: bytes,
    producer_role: str,
) -> dict[str, Any]:
    if not producer_role.strip():
        raise AuthenticationError("Producer-declared role is required")
    key = load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise AuthenticationError("Only Ed25519 private keys are supported")
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    fingerprint = sha256(public).hexdigest()
    statement = {
        "canonicalization": "mun-canonical-json-v1",
        "artifact_schema_version": artifact.get("schema_version"),
        "artifact_sha256": _digest(artifact),
        "receipt_sha256": _digest(receipt),
        "producer_role": producer_role,
        "public_key": base64.b64encode(public).decode("ascii"),
        "public_key_fingerprint": fingerprint,
    }
    return {
        "schema_version": 1,
        "envelope_kind": "mun-producer-authentication",
        "algorithm": "Ed25519",
        **statement,
        "signature": base64.b64encode(key.sign(canonical_json_bytes(statement))).decode("ascii"),
        "claims": {"accuracy": False, "reviewer_identity": False, "consent": False, "authorization": False},
    }


def verify_artifact(
    artifact: Mapping[str, Any],
    receipt: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    if envelope.get("schema_version") != 1 or envelope.get("canonicalization") != "mun-canonical-json-v1":
        raise AuthenticationError("Ambiguous canonicalization or unsupported envelope schema")
    if envelope.get("algorithm") != "Ed25519":
        raise AuthenticationError("Unknown authentication algorithm")
    if envelope.get("artifact_schema_version") != artifact.get("schema_version"):
        raise AuthenticationError("Artifact schema version mismatch")
    if envelope.get("artifact_sha256") != _digest(artifact):
        raise AuthenticationError("Authenticated artifact content was altered")
    if envelope.get("receipt_sha256") != _digest(receipt):
        raise AuthenticationError("Authenticated receipt was altered")
    try:
        public_bytes = base64.b64decode(str(envelope["public_key"]), validate=True)
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Malformed public key or signature") from exc
    fingerprint = sha256(public_bytes).hexdigest()
    if envelope.get("public_key_fingerprint") != fingerprint:
        raise AuthenticationError("Public key fingerprint mismatch")
    statement = {
        key: envelope[key]
        for key in (
            "canonicalization", "artifact_schema_version", "artifact_sha256", "receipt_sha256",
            "producer_role", "public_key", "public_key_fingerprint",
        )
    }
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, canonical_json_bytes(statement))
    except (ValueError, InvalidSignature, KeyError) as exc:
        raise AuthenticationError("Malformed or invalid producer signature") from exc
    return {
        "valid": True,
        "algorithm": "Ed25519",
        "artifact_sha256": envelope["artifact_sha256"],
        "public_key_fingerprint": fingerprint,
        "producer_role": envelope["producer_role"],
        "claims": {"accuracy": False, "reviewer_identity": False, "consent": False, "authorization": False},
    }
