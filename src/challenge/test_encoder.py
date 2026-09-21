from src.challenge.encoder import decode_events, encode_flag


FLAG = "PROVUE{TEST}"


def main():
    encoded = encode_flag(FLAG)

    print("flag:", encoded["flag"])
    print("symbols:", encoded["symbols"])
    print("events:", encoded["events"])

    decoded = decode_events(encoded["events"])

    print("decoded symbols:", decoded["symbols"])
    print("decoded flag:", decoded["flag"])

    assert decoded["flag"] == FLAG

    print("\nround_trip=PASS")


if __name__ == "__main__":
    main()