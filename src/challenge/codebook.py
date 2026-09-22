CHANNEL_STEPS = [40, 50, 60, 70]
NUM_CHANNELS = len(CHANNEL_STEPS)
SYMBOLS = range(16)

CODEBOOK = {symbol: tuple(CHANNEL_STEPS[i] for i in range(NUM_CHANNELS) if symbol & (1 << i)) for symbol in SYMBOLS}
REVERSE_CODEBOOK = {events: symbol for symbol, events in CODEBOOK.items()}