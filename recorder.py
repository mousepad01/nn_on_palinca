import os
from struct import Struct
from time import sleep

DATA_PATH = "timestamps.bin"
FLUSH_THRESH = 30 # pressed keys

DEVICES_PATH = "/proc/bus/input/devices"
KEYBOARD_PATH = "/dev/input/"

EV_SYN = 0x00
EV_KEY = 0x01
EV_REP = 0x14

input_event = Struct("llHHi")

def detect_possible_keyboards():
    """parses /proc/bus/input/devices to find
        candidates for keyboard"""
    
    with open(DEVICES_PATH, "rb") as devices:
        
        # candidates
        devices_list = []

        devices_desc = devices.read(2 ** 16).decode().split("\n\n")
        
        for d in devices_desc:

            try:
            
                if "EV" not in d:
                    continue

                if "Handlers" not in d:
                    continue

                event_types = d.split("EV")[1].split("\n")[0]
                event_types = int(event_types[1:], 16)

                if event_types & (1 << EV_SYN) == 0 or \
                    event_types & (1 << EV_KEY) == 0 or \
                    event_types & (1 << EV_REP) == 0:

                    continue

                handlers = d.split("Handlers")[1].split("\n")[0]

                for h in handlers[1:].split():

                    if "event" in h:
                        devices_list.append(h)
                        break

            except Exception:
                continue

        return devices_list

def filter_candidates(devices_list):

    def _read(f):

        b = os.read(f, input_event.size)
        return input_event.unpack(b) if b is not None else None

    print("[!] Please run as SUPERUSER, "
            "otherwise the script will not be able to run due to security concerns")
    sleep(2)

    print("[!] You are going to be repeatedly prompted to press any key"
            + " within a timeframe of a few seconds, "
            + "to check whether this is the currently used keyboard...\n\n")

    sleep(5)

    for device in devices_list:

        dev_path = f"{KEYBOARD_PATH}{device}"
        keyboard = os.open(dev_path, os.O_RDONLY | os.O_NONBLOCK)

        print("[!] Spam any key")

        i = 0
        while i < 1500000:

            try:

                event = _read(keyboard)
                if event is not None and event[2] == EV_KEY:

                    print(f"[*] Ok, found it: {dev_path}")
                    print(f"[*] (you can stop spamming)")
                    return dev_path

            except Exception:
                i += 1

        print(f"[!] Not this one ({dev_path})")
        sleep(1)

    raise Exception("Cannot find keyboard. Terminated")

def readloop(keyboard_path, data_path):

    # BLOCKING
    def _read(f):
        return input_event.unpack(f.read(input_event.size))

    print(f"[*] Data will be collected in this dir, in the binary file '{data_path}'")
    print(f"[*] The script will shortly start collecting key timestamps (and only TIMESTAMPS)")
    print(f"[*] To stop this, execute 'sudo kill -9 {os.getpid()}'")

    sleep(2)

    with open(keyboard_path, "rb") as keyboard:
        with open(data_path, "ab+") as data:
            
            idx = 0
            while True:
                
                ev = _read(keyboard)
                
                if ev[2] != EV_KEY or ev[4] != 0:
                    # only key presses, because it also records key releases etc
                    continue

                # NOTE uncomment to see the output printed
                # print(ev[0], ev[1])
                
                # (seconds since epoch, miliseconds) 
                # (check input_event struct /include/linux/input.h)
                data.write(int.to_bytes(ev[0], 8, 'little') + int.to_bytes(ev[1], 8, 'little'))
              
                idx += 1
                if idx == FLUSH_THRESH:
                    data.flush()
                    idx = 0

if __name__ == "__main__":

    devices_list = detect_possible_keyboards()
    keyboard_path = filter_candidates(devices_list)

    readloop(keyboard_path, DATA_PATH)