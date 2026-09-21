from src.challenge.codebook import CODEBOOK, REVERSE_CODEBOOK


def flag_to_symbols(flag):
    data = flag.encode("utf-8")
    symbols = []

    for byte in data:
        symbols.append(byte >> 4)
        symbols.append(byte & 0x0F)

    return symbols


def symbols_to_flag(symbols):
    if len(symbols) % 2 != 0:
        raise ValueError("Symbol sequence must contain an even number of symbols.")

    data = bytearray()

    for index in range(0, len(symbols), 2):
        high = symbols[index]
        low = symbols[index + 1]

        if not 0 <= high <= 15 or not 0 <= low <= 15:
            raise ValueError("Symbols must be in range 0..15.")

        data.append((high << 4) | low)

    return data.decode("utf-8")


def symbols_to_events(symbols):
    return [CODEBOOK[symbol] for symbol in symbols]


def events_to_symbols(events):
    try:
        return [REVERSE_CODEBOOK[tuple(event)] for event in events]
    except KeyError as exc:
        raise ValueError(f"Unknown temporal event: {exc.args[0]}") from exc


def encode_flag(flag):
    symbols = flag_to_symbols(flag)
    events = symbols_to_events(symbols)

    return {
        "flag": flag,
        "symbols": symbols,
        "events": events,
    }


def decode_events(events):
    symbols = events_to_symbols(events)
    flag = symbols_to_flag(symbols)

    return {
        "symbols": symbols,
        "flag": flag,
    }