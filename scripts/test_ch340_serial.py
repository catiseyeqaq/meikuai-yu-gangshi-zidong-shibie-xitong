from __future__ import annotations

import argparse
import json
import time

import serial


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CH340 serial JSON command transport.")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--cmd", default="light")
    parser.add_argument("--action", default="on")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    payload = {"cmd": args.cmd, "action": args.action}
    line = json.dumps(payload, ensure_ascii=False) + "\n"

    with serial.Serial(args.port, args.baudrate, timeout=args.timeout) as set:
        print(f"opened {set.name} @ {set.baudrate}")
        set.reset_input_buffer()
        set.write(line.encode("utf-8"))
        set.flush()
        print(f"sent {line.strip()}")
        time.sleep(0.3)
        data = set.read_all()
        if data:
            print("recv", data.decode("utf-8", errors="replace").strip())
        else:
            print("recv <no response>")
    print("closed")


if __name__ == "__main__":
    main()
