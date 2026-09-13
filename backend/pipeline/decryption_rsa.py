"""
decryption_rsa.py — RSA decryption side.

RSA is symmetric in structure (the same modular-exponentiation routine
performs both directions, differing only in which exponent is used),
so the core `rsa_decrypt` function already lives in encryption_rsa.py
per the project spec ("RSA decrypt already covered in
encryption_rsa.py's rsa_decrypt"). This module re-exports the
decrypt-side functions under decryption_rsa's own namespace so callers
that conceptually want "the decryption module" have a stable import
path, and adds a thin report wrapper for the API layer.
"""

from __future__ import annotations
from .encryption_rsa import rsa_decrypt, unwrap_aes_key, mod_exp


def rsa_decrypt_report(c: int, private_key: dict) -> dict:
    """Convenience wrapper producing an API response payload for RSA decryption."""
    m = rsa_decrypt(c, private_key)
    return {"ciphertext_int": c, "recovered_message_int": m}
