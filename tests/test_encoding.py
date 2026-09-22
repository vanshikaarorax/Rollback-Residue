from src2.encoding import byte_to_bits, bits_to_byte


def test_roundtrip():
    for value in range(256):
        bits = byte_to_bits(value)
        assert bits_to_byte(bits) == value