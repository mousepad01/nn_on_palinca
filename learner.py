from enum import Enum
import random

class DataState(Enum):
    UNINITIALIZED = "uninitialized"
    RAW = "raw"
    READY = "ready"

class HumanVsRnd:
    """Its task is to differentiate
        between human keystroke timestamps
        and random keystroke timestamps"""

    def __init__(self, data_path = "timestamps.bin",
                        new_timeslice_thresh = 5,
                        micros_to_ms = 1000,
                        timeslice_len = 10,
                        random_to_human_ratio = 1.5,
                        random_min_decal = 200,
                        random_max_decal = 2000,
                        ):

        assert(1 <= micros_to_ms <= 1000000)
        assert(new_timeslice_thresh > 0)
        assert(timeslice_len > 0)
        assert(random_to_human_ratio > 0) 
        assert(random_min_decal > 0)
        assert(random_max_decal < new_timeslice_thresh * micros_to_ms)

        self.timeslice_len = timeslice_len
        """timeslice length to be fed to the ML model"""

        self.micros_to_ms = micros_to_ms
        """conversion between microseconds (raw_data[idx][1]) to 'miliseconds'"""

        self.data_path = data_path
        self.new_timeslice_thresh = new_timeslice_thresh
        """threshold (in seconds) that forces a new timeslice"""

        self.data_state = DataState.UNINITIALIZED
        self.data = None
        """human data"""

        self.random_data_state = DataState.UNINITIALIZED
        self.random_data = None
        """random data"""

        self.random_min_decal = random_min_decal
        """minimum milisec distance between random keystrokes"""
        self.random_max_decal = random_max_decal
        """maximum milisec distance between random keystrokes"""

        self.random_to_human_ratio = random_to_human_ratio
        """ratio of random vs human data"""

    def load_data(self):

        data_buf = []

        with open(self.data_path, "rb") as data:
            alldata = data.read()

        last_s, last_ms = 0, 0

        idx = 0
        while idx < len(alldata):

            s = int.from_bytes(alldata[idx: idx + 8], 'little')
            ms = int.from_bytes(alldata[idx + 8: idx + 16], 'little')

            assert((s, ms) > (last_s, last_ms))

            data_buf.append((s, ms))

            # print(f"{idx // 16}: {s}, {ms}")

            idx += 16

        self.data = data_buf
        self.data_state = DataState.RAW

    def random_data_gen(self):
        """generate 'random' keystroke timestamps"""

        assert(self.data_state is DataState.READY)
        
        random_data_len = self.random_to_human_ratio
        self.random_data = []

        for _ in range(random_data_len):
            
            self.random_data.append([0])
            self.t_offset = 0

            for _ in range(self.timeslice_len):
                
                self.t_offset += random.randint(self.random_min_decal, self.random_max_decal)
                self.random_data[-1].append(self.t_offset)

        self.random_data_state = DataState.READY
    
    def format_data(self):
        """* make timestamps relative to one another
            * split in different timeslices"""

        raw_data = self.data

        data_timeslices = []

        last_t = 0

        current_timeslice_base_s = 0
        current_timeslice_base_ms = 0
        current_timeslice_len = 0
        
        idx = 0
        timeslice_idx = -1

        while idx < len(raw_data):

            if current_timeslice_len == self.timeslice_len + 1 or \
                raw_data[idx][0] - last_t > self.new_timeslice_thresh:
                
                timeslice_idx += 1
                data_timeslices.append([])

                current_timeslice_base_s = raw_data[idx][0]
                current_timeslice_base_ms = raw_data[idx][1] // self.micros_to_ms
                current_timeslice_len = 0

            data_timeslices[-1].append((raw_data[idx][0] - current_timeslice_base_s) * self.micros_to_ms + 
                                    raw_data[idx][1] // self.micros_to_ms - current_timeslice_base_ms)

            assert(data_timeslices[-1][-1] >= 0)

            current_timeslice_len += 1
            last_t = raw_data[idx][0]
            idx += 1

        self.data = []
        for tslice in data_timeslices:

            assert(len(tslice) <= self.timeslice_len + 1)

            if len(tslice) == self.timeslice_len + 1:
                self.data.append(tslice[1:])    # first entry always 0

        self.data_state = DataState.READY
        
        # print(len(self.data))

if __name__ == "__main__":

    model = HumanVsRnd()

    model.load_data()
    model.format_data()

