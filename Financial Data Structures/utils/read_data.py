from datetime import datetime
import pytz
from tqdm import tqdm


def read_ticks(csv, nrows=None, date_handling=None):
    """
    Reads a csv file with the following columns:
    - time: timestamp in isoformat
    - price: float
    - volume: float
    - symbol: str

    Args:
    - csv: str, path to the csv file
    - nrows: int, number of rows to read
    - dates: how to handle timestamps. Options are:
        - None: no processing (returns isoformat)
        - "epoch": returns epoch time
        - "datetime": returns datetime object in US/Eastern timezone
    """

    def date_processing(dates_arg):
        if dates_arg == "epoch":
            return lambda x: datetime.fromisoformat(x).timestamp()
        if dates_arg == "datetime":
            return lambda x: datetime.fromisoformat(x).replace(tzinfo=pytz.utc).astimezone(pytz.timezone("US/Eastern"))
        return lambda x: x

    process_date = date_processing(date_handling)

    data = {"time": [], "price": [], "volume": [], "symbol": []}
    with open(csv, "r") as f:
        for i, line in enumerate(tqdm(f, total=82971690)):
            if i == 0:
                continue
            if nrows is not None and i > nrows:
                break
            line = line.strip().split(",")
            data["time"].append(process_date(line[0]))
            data["price"].append(float(line[1]))
            data["volume"].append(float(line[2]))
            data["symbol"].append(line[3])
    return data
