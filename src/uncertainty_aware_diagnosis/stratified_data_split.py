import polars as pl  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from typing import Optional, List  # type: ignore


def split_and_save_csv(
    input_csv: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify_col: Optional[str] = None,
    min_class_count: int = 25,
    random_state: int = 42,
):
    """
    Read `input_csv`, split into train/val/test, and write out three files.

    Parameters
    ----------
    input_csv : str
        Path to the source CSV.
    train_csv : str
        Path where the train split CSV will be saved.
    val_csv : str
        Path where the validation split CSV will be saved.
    test_csv : str
        Path where the test split CSV will be saved.
    train_frac : float, default=0.7
        Fraction of the data for the training set.
    val_frac : float, default=0.15
        Fraction for the validation set.
    test_frac : float, default=0.15
        Fraction for the test set.
    stratify_col : str, optional
        Column name to use for stratified splitting. If None, splits randomly.
    random_state : int, default=42
        RNG seed for reproducibility.
    """
    # sanity check
    total = train_frac + val_frac + test_frac
    assert abs(total - 1.0) < 1e-6, "train_frac+val_frac+test_frac must sum to 1"

    # 1) Load full dataset
    df = pl.read_csv(input_csv).to_pandas()

    # 2) Drop classes with fewer than min_class_count samples
    if stratify_col is not None:
        class_counts = df[stratify_col].value_counts()
        valid_classes = class_counts[class_counts >= min_class_count].index
        df = df[df[stratify_col].isin(valid_classes)]

    # 3) First split: train vs temp (val+test)
    stratify_vals = df[stratify_col] if stratify_col else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=stratify_vals,
        random_state=random_state,
        shuffle=True,
    )

    # 4) Second split: temp → val + test
    #    Adjust val fraction relative to temp_df size
    val_relative = val_frac / (val_frac + test_frac)
    stratify_temp = temp_df[stratify_col] if stratify_col else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        stratify=stratify_temp,
        random_state=random_state,
        shuffle=True,
    )

    # 5) Write out CSVs
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(
        f"Done! Splits sizes: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )


if __name__ == "__main__":
    # split data into train, val, test
    input_csv = "./data/synthetic_noisy_admission_data.csv"
    train_csv = "./data/synthetic_train.csv"
    val_csv = "./data/synthetic_val.csv"
    test_csv = "./data/synthetic_test.csv"
    target = "admission_code"

    split_and_save_csv(
        input_csv=input_csv,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        stratify_col=target,  # preserve class balance
        min_class_count=25,
        random_state=42,
    )
