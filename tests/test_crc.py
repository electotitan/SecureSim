import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import error_detection_crc as crc


def test_known_test_vector_123456789():
    # The canonical CRC-32 check value published in the CRC catalogue.
    assert crc.crc32_compute(b"123456789") == 0xCBF43926


def test_empty_input():
    assert crc.crc32_compute(b"") == 0x00000000


def test_bitwise_reference_matches_table_driven():
    import random
    random.seed(21)
    for _ in range(20):
        n = random.randint(0, 200)
        data = bytes(random.randint(0, 255) for _ in range(n))
        assert crc.crc32_compute(data) == crc.crc32_bitwise_reference(data)


def test_verify_detects_no_corruption():
    data = b"the quick brown fox"
    checksum = crc.crc32_compute(data)
    assert crc.crc32_verify(data, checksum) is True


def test_verify_detects_single_bit_flip():
    data = bytearray(b"the quick brown fox")
    checksum = crc.crc32_compute(bytes(data))
    data[3] ^= 0x01  # flip one bit
    assert crc.crc32_verify(bytes(data), checksum) is False


def test_table_has_256_entries():
    assert len(crc.CRC32_TABLE) == 256
