"""Stage 11 (RSA part): RSA decryption and symmetric-key unwrapping.

Reuses the modular arithmetic of ``encryption_rsa`` (square-and-multiply, no built-in three-argument
``pow``).  Deliberately independent of the AES modules: in this simulator RSA only plays the role of
the educational *key exchange*, and the AES message path never depends on it.

EDUCATIONAL AND INSECURE (textbook RSA, see ``encryption_rsa``).
"""
from __future__ import annotations

from typing import Any

from backend.pipeline.encryption_rsa import (
    PrivateKey, _check_message, int_to_key, mod_exp_trace, rsa_decrypt,
)

__all__ = ["rsa_decrypt", "rsa_decrypt_trace", "rsa_unwrap_key"]


def rsa_decrypt_trace(c: int, priv_key: PrivateKey) -> dict[str, Any]:
    """Decryption with the full square-and-multiply trace (integers, not strings)."""
    if not isinstance(priv_key, PrivateKey):
        raise TypeError("priv_key must be a PrivateKey")
    _check_message(c, priv_key.n, "c")
    return mod_exp_trace(c, priv_key.d, priv_key.n)


def rsa_unwrap_key(ciphertext: int, priv_key: PrivateKey, key_length: int) -> bytes:
    """Decrypt a wrapped symmetric key of ``key_length`` bytes.

    Raises ValueError if the decrypted integer does not fit in ``key_length`` bytes (which happens,
    for example, when the wrong private key is used).
    """
    return int_to_key(rsa_decrypt(ciphertext, priv_key), key_length)
