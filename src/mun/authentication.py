from __future__ import annotations

import base64
import binascii
from hashlib import sha256
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_private_key

from .artifacts import canonical_json_bytes
from .errors import MunError


class AuthenticationError(MunError):
    def __init__(self, message: str, code: str = "authentication_invalid") -> None:
        super().__init__(message)
        self.code = code


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
    if not isinstance(artifact, Mapping):
        raise AuthenticationError("Artifact must be a JSON object", "artifact_malformed")
    if not isinstance(receipt, Mapping):
        raise AuthenticationError("Receipt must be a JSON object", "receipt_malformed")
    if not isinstance(envelope, Mapping):
        raise AuthenticationError("Authentication envelope must be a JSON object", "envelope_malformed")
    required = {
        "schema_version", "envelope_kind", "algorithm", "canonicalization", "artifact_schema_version",
        "artifact_sha256", "receipt_sha256", "producer_role", "public_key", "public_key_fingerprint",
        "signature", "claims",
    }
    if set(envelope) != required:
        raise AuthenticationError("Authentication envelope fields are missing or ambiguous", "envelope_fields")
    if envelope.get("envelope_kind") != "mun-producer-authentication":
        raise AuthenticationError("Unsupported authentication envelope kind", "envelope_kind")
    if envelope.get("schema_version") != 1 or envelope.get("canonicalization") != "mun-canonical-json-v1":
        raise AuthenticationError("Ambiguous canonicalization or unsupported envelope schema", "canonicalization")
    if envelope.get("algorithm") != "Ed25519":
        raise AuthenticationError("Unknown authentication algorithm", "algorithm")
    claims = envelope.get("claims")
    expected_claims = {"accuracy": False, "reviewer_identity": False, "consent": False, "authorization": False}
    if claims != expected_claims:
        raise AuthenticationError("Authentication claims are malformed or ambiguous", "claims")
    for field in ("artifact_sha256", "receipt_sha256", "public_key_fingerprint"):
        value = envelope.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise AuthenticationError("Authentication digest fields are malformed", "envelope_fields")
    if not isinstance(envelope.get("producer_role"), str) or not envelope["producer_role"].strip():
        raise AuthenticationError("Producer-declared role is malformed", "producer_role")
    if envelope.get("artifact_schema_version") != artifact.get("schema_version"):
        raise AuthenticationError("Artifact schema version mismatch", "artifact_schema")
    if envelope.get("artifact_sha256") != _digest(artifact):
        raise AuthenticationError("Authenticated artifact content was altered", "artifact_altered")
    if envelope.get("receipt_sha256") != _digest(receipt):
        raise AuthenticationError("Authenticated receipt was altered", "receipt_altered")
    try:
        if not isinstance(envelope["public_key"], str) or not isinstance(envelope["signature"], str):
            raise ValueError
        public_bytes = base64.b64decode(envelope["public_key"], validate=True)
        signature = base64.b64decode(envelope["signature"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AuthenticationError("Malformed public key or signature", "encoding") from exc
    if len(public_bytes) != 32 or len(signature) != 64:
        raise AuthenticationError("Malformed public key or signature", "encoding")
    fingerprint = sha256(public_bytes).hexdigest()
    if envelope.get("public_key_fingerprint") != fingerprint:
        raise AuthenticationError("Public key fingerprint mismatch", "fingerprint")
    statement = {
        key: envelope[key]
        for key in (
            "canonicalization", "artifact_schema_version", "artifact_sha256", "receipt_sha256",
            "producer_role", "public_key", "public_key_fingerprint",
        )
    }
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, canonical_json_bytes(statement))
    except (ValueError, InvalidSignature) as exc:
        raise AuthenticationError("Malformed or invalid producer signature", "signature") from exc
    return {
        "valid": True,
        "algorithm": "Ed25519",
        "artifact_sha256": envelope["artifact_sha256"],
        "public_key_fingerprint": fingerprint,
        "producer_role": envelope["producer_role"],
        "claims": {"accuracy": False, "reviewer_identity": False, "consent": False, "authorization": False},
    }
