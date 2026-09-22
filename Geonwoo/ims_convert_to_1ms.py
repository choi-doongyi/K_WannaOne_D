from pathlib import Path
from datetime import datetime, timedelta
import csv

import numpy as np


RAW_ROOT = Path("/private/tmp/ims_raw")
OUTPUT_ROOT = Path("data/ims_1ms")
SAMPLE_RATE_HZ = 20_000
TARGET_RATE_HZ = 1_000
GROUP_SIZE = SAMPLE_RATE_HZ // TARGET_RATE_HZ


def convert_experiment(experiment: str, output_name: str) -> None:
    files = sorted(
        (RAW_ROOT / experiment).rglob("*"),
        key=lambda path: path.name,
    )
    files = [path for path in files if path.is_file() and path.name[0].isdigit()]

    output_file = OUTPUT_ROOT / output_name
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "channel_1", "channel_2", "channel_3", "channel_4",
            "channel_5", "channel_6", "channel_7", "channel_8",
        ])

        for index, source_file in enumerate(files, start=1):
            values = np.loadtxt(source_file, dtype=np.float32)
            usable = (len(values) // GROUP_SIZE) * GROUP_SIZE
            values = values[:usable]
            values = values.reshape(-1, GROUP_SIZE, 8).mean(axis=1)

            start_time = datetime.strptime(
                source_file.name,
                "%Y.%m.%d.%H.%M.%S",
            )

            for row_index, row in enumerate(values):
                timestamp = start_time + timedelta(milliseconds=row_index)
                writer.writerow([
                    timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    *row.tolist(),
                ])

            if index % 100 == 0 or index == len(files):
                print(f"{experiment}: {index}/{len(files)}")

    print(f"완료: {output_file}")


convert_experiment("1st_test", "test_set_1_1ms.csv")
convert_experiment("2nd_test", "test_set_2_1ms.csv")
convert_experiment("3rd_test", "test_set_3_1ms.csv")
