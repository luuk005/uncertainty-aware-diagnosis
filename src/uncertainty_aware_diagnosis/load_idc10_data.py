import polars as pl  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # type: ignore
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
        numerical (List[str], optional): List of numeric column names.
        categorical (List[str], optional): List of low-cardinality categorical column names.
        high_card (List[str], optional): List containing the name of the high-cardinality category
        target (str, optional): Name of the target column.
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
        categorical: List[str] = ["gender","clinical_specialty"],
        high_card: List[str] = ["procedure"],
        target: str = "icd10_principal_diagnosis",
        dropna: bool = True,
        use_embedding: bool = False,
        ohe_categories: Optional[List[List[str]]] = None,
        scaler: Optional[MinMaxScaler] = None,
        encoder: Optional[OneHotEncoder] = None,
    ):
        # 1) Load data via Polars for speed, then to pandas
        df = pl.read_csv(csv_path).to_pandas()
        if dropna:
            df = df.dropna()
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

