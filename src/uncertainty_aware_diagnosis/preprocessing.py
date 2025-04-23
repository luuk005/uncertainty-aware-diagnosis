import polars as pl
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np


class AIOCdata(Dataset):
    def __init__(
        self,
        csv_path: str,
        numerical: list = ['age'],
        categorical: list = ["gender", "clinical_specialty", "dbc_specialty_code", 
                              "dbc_diagnosis_code", "icd10_subtraject_code"],
        high_card: list = ["procedure_code"],
        target: str = "icd10_main_code",
        use_embedding: bool = True,
        ohe_categories: list = None,   # Optional: pass a list of category lists
        scaler: MinMaxScaler = None,    # Optional: pass a fitted scaler
        encoder: OneHotEncoder = None   # Optional: pass a fitted encoder
    ):
        """
        csv_path      : path to your CSV file
        numerical     : list of numeric column names
        categorical   : list of low-cardinality categorical names
        high_card     : list with your 1 high-cardinality category name (e.g. ['procedure_code'])
        target        : name of the target column
        use_embedding : if True, procedure_code → nn.Embedding; otherwise it's dropped
        ohe_categories: if provided, a list of lists of all categories for each cat col
        scaler/encoder: if provided, use these instead of fitting new ones
        """
        # 1) Load data via Polars for speed, then to pandas
        df = pl.read_csv(csv_path).to_pandas()
        # 2) Ensure correct dtype
        for col in categorical + high_card + [target]:
            df[col] = df[col].astype('category')
        
        self.numerical   = numerical
        self.categorical = categorical
        self.high_card   = high_card
        self.target      = target
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
                categories=cats,
                sparse_output=False,
                handle_unknown="error"
            )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.scaler, numerical),
                ("cat", self.encoder, categorical),
            ],
            remainder="drop"
        )
        
        # 4) Fit & transform X_small (numeric + low-card cats)
        X_small_np = self.preprocessor.fit_transform(df[numerical + categorical])
        self.X = torch.from_numpy(X_small_np.astype('float32'))
        
        # 5) Extract / encode high-card feature if needed
        if use_embedding:
            # Handle high-card features: convert each to codes and stack
            codes_list = [df[col].cat.codes.values for col in high_card]
            codes_np = np.stack(codes_list, axis=1)  # shape: (n_samples, n_high_card)
            self.high_card_codes = torch.from_numpy(codes_np.astype('int64')) #first feature codes = self.high_card_codes[:, 0]
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
        self.y      = torch.from_numpy(y_codes.astype('int64'))
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
        if self.use_embedding:
            hc = self.high_card_codes[idx]  # Tensor of shape [n_high_card]
            return x, hc, y
        else:
            return x, y


def split_and_save_csv(
    input_csv: str,
    train_csv: str,
    val_csv:   str,
    test_csv:  str,
    train_frac: float = 0.7,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    stratify_col: str = None,
    random_state: int = 42
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
        shuffle=True
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
        shuffle=True
    )
    
    # 4) Write out CSVs
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)
    test_df.to_csv(test_csv,    index=False)
    
    print(f"Done! Splits sizes: "
          f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
if __name__ == "__main__":
    # split data
    input_csv="./data/corrupted_synthetic_data.csv"
    train_csv="./data/train.csv"
    val_csv="./data/val.csv"
    test_csv="./data/test.csv"
    target="icd10_main_code"

    split_and_save_csv(
        input_csv=input_csv,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        stratify_col=target,  # preserve class balance
        random_state=42
    )
