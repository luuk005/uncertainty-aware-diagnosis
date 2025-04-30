import polars as pl  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, List  # type: ignore


class ICD10data(Dataset):
    """Dataset class for ICD10 data preprocessing.

    This class handles the loading, preprocessing, and encoding of ICD10 data from a CSV file.
    It supports both numerical and categorical features, including high-cardinality categorical features
    that can be embedded into a neural network.

    Args:
        csv_path (str): Path to the CSV file containing the data.
        numerical (List[str], optional): List of numeric column names. Defaults to ["age"].
        categorical (List[str], optional): List of low-cardinality categorical column names.
            Defaults to a predefined list of specialty and diagnosis codes.
        high_card (List[str], optional): List containing the name of the high-cardinality category
            (e.g., ['procedure_code']). Defaults to ["procedure_code"].
        target (str, optional): Name of the target column. Defaults to "icd10_main_code".
        use_embedding (bool, optional): If True, the procedure_code will be converted to an
            embedding; otherwise, it will be dropped. Defaults to False.
        ohe_categories (Optional[List[List[str]]], optional): If provided, a list of lists of all
            categories for each categorical column. Defaults to None.
        scaler (Optional[MinMaxScaler], optional): If provided, use this scaler instead of fitting
            a new one. Defaults to None.
        encoder (Optional[OneHotEncoder], optional): If provided, use this encoder instead of
            fitting a new one. Defaults to None.

    Attributes:
        numerical (List[str]): List of numeric column names.
        categorical (List[str]): List of low-cardinality categorical column names.
        high_card (List[str]): List of high-cardinality categorical column names.
        target (str): Name of the target column.
        use_embedding (bool): Indicates if embedding is used for high-cardinality features.
        scaler (MinMaxScaler): Scaler for numerical features.
        encoder (OneHotEncoder): Encoder for categorical features.
        preprocessor (ColumnTransformer): Preprocessing pipeline for the dataset.
        X (torch.Tensor): Processed feature tensor.
        high_card_codes (Optional[torch.Tensor]): Encoded high-cardinality feature tensor.
        num_codes_per_feature (Optional[List[int]]): Cardinalities for embedding layers.
        y (torch.Tensor): Encoded target tensor.
        classes (np.ndarray): Unique classes for the target variable.
    """

    def __init__(
        self,
        csv_path: str,
        numerical: List[str] = ["age"],
        categorical: List[str] = [
            "gender",
            "clinical_specialty",
            "dbc_specialty_code",
            "dbc_diagnosis_code",
            "icd10_subtraject_code",
        ],
        high_card: List[str] = ["procedure_code"],
        target: str = "icd10_main_code",
        use_embedding: bool = False,
        ohe_categories: Optional[List[List[str]]] = None,
        scaler: Optional[MinMaxScaler] = None,
        encoder: Optional[OneHotEncoder] = None,
    ):
        # 1) Load data via Polars for speed, then to pandas
        df = pl.read_csv(csv_path).to_pandas()
        # 2) Ensure correct dtype
        for col in categorical + high_card + [target]:
            df[col] = df[col].astype("category")

        self.numerical = numerical
        self.categorical = categorical
        self.high_card = high_card
        self.target = target
        self.use_embedding = use_embedding

        # 3) Build / reuse preprocessing objects
        #    a) MinMax for numericals
        self.scaler = scaler or MinMaxScaler()
        #    b) OneHot for low-card cats
        if encoder and ohe_categories:
            self.encoder = encoder
        else:
            # infer categories from data if not given
            cats = ohe_categories or [df[col].cat.categories for col in categorical]
            self.encoder = OneHotEncoder(
                categories=cats, sparse_output=False, handle_unknown="error"
            )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.scaler, numerical),
                ("cat", self.encoder, categorical),
            ],
            remainder="drop",
        )

        # 4) Fit & transform X_small (numeric + low-card cats)
        X_small_np = self.preprocessor.fit_transform(df[numerical + categorical])
        self.X = torch.from_numpy(X_small_np.astype("float32"))

        # 5) Extract / encode high-card feature if needed
        if use_embedding:
            # Handle high-card features: convert each to codes and stack
            codes_list = [df[col].cat.codes.values for col in high_card]
            codes_np = np.stack(codes_list, axis=1)  # shape: (n_samples, n_high_card)
            self.high_card_codes = torch.from_numpy(
                codes_np.astype("int64")
            )  # first feature codes = self.high_card_codes[:, 0]
            # record cardinalities for embedding layers
            self.num_codes_per_feature = [
                df[col].cat.categories.size for col in high_card
            ]

        else:
            # not used
            self.high_card_codes = None
            self.num_codes_per_feature = None

        # 6) Label-encode target
        self.le = LabelEncoder()
        y_codes = self.le.fit_transform(df[target])
        self.y = torch.from_numpy(y_codes.astype("int64"))
        self.classes = self.le.classes_  # handy for mapping back

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """
        Returns:
          if use_embedding:
            (X_tensor, hc_code_index, target_label)
          else:
            (X_tensor, target_label)
        """
        x = self.X[idx]
        y = self.y[idx]
        if self.use_embedding and self.high_card_codes is not None:
            hc = self.high_card_codes[idx]  # Tensor of shape [n_high_card]
            return x, hc, y
        else:
            return x, y


def split_and_save_csv(
    input_csv: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify_col: Optional[str] = None,
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

    # 2) First split: train vs temp (val+test)
    stratify_vals = df[stratify_col] if stratify_col else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=stratify_vals,
        random_state=random_state,
        shuffle=True,
    )

    # 3) Second split: temp → val + test
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

    # 4) Write out CSVs
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(
        f"Done! Splits sizes: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )


if __name__ == "__main__":
    # split data into train, val, test
    input_csv = "./data/corrupted_synthetic_data.csv"
    train_csv = "./data/train.csv"
    val_csv = "./data/val.csv"
    test_csv = "./data/test.csv"
    target = "icd10_main_code"

    split_and_save_csv(
        input_csv=input_csv,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        stratify_col=target,  # preserve class balance
        random_state=42,
    )
