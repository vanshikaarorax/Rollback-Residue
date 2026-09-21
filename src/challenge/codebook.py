BASE_STEP = 40
GAPS = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136]

CODEBOOK = {symbol: (BASE_STEP, BASE_STEP + gap) for symbol, gap in enumerate(GAPS)}

REVERSE_CODEBOOK = {timing: symbol for symbol, timing in CODEBOOK.items()}