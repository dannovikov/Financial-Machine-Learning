import pandas as pd  # pylint: disable=missing-module-docstring


class Prices:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.rename(
            {"Adj Close": "price", "Date": "date", "Volume": "volume"},
            axis=1,
            inplace=True,
        )
        self.data = df[["date", "price", "volume"]]
        self.iter_index = 0  # used in `for` loops

    def __getitem__(self, index):
        return self.data["price"].iloc[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < len(self.data):
            result = self.data["price"].iloc[self.iter_index]
            self.iter_index += 1
            return result
        else:
            raise StopIteration

    def __repr__(self):
        return repr(self.data)


if __name__ == "__main__":
    prices = Prices("../../Historical Data/NVDA.csv")
    for i in prices:
        print(i)
