import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("train/surrogate_data.csv")

    df["gpu_hours"] = df["train_duration(s)"] * df["world_size"] / 3600
    result = df.groupby("train_gpu_name")["gpu_hours"].sum()
    print(result)
