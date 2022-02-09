DATA_PATH = "timestamps.bin"

# FIXME
def load_data(data_path):

    data_buf = []

    with open(data_path, "rb") as data:
        alldata = data.read()

    last_s, last_ms = 0, 0

    idx = 0
    while idx < len(alldata):

        s = int.from_bytes(alldata[idx: idx + 8], 'little')
        ms = int.from_bytes(alldata[idx + 8: idx + 16], 'little')

        assert((s, ms) > (last_s, last_ms))

        data_buf.append((s, ms))

        print(f"{idx // 16}: {s}, {ms}")

        idx += 16

    #print(data_buf)

if __name__ == "__main__":

    load_data(DATA_PATH)