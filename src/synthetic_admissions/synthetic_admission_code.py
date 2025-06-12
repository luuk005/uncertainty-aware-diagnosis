import numpy as np
import polars as pl
import string

# Constants
LETTERS_A_J = list(string.ascii_uppercase[:10])  # A-J
LETTERS_K_VZ = list(string.ascii_uppercase[10:22]) + ['Z']  # K-V + Z
ALL_LETTERS = LETTERS_A_J + LETTERS_K_VZ


def generate_synthetic_admissions(
    n_samples=10000,
    p=0.6,
    variance=0.025,
    p2=0.5,
    corr=0.89,
    corr2=0.78,
    hospital_specialization='cardiology',
    seed=42
):
    """
    Generate synthetic hospital admission data.

    Parameters:
    - n_samples: Number of samples to generate
    - p: Total probability mass for letters A-J
    - variance: Variance in the Gaussian noise for A-J probabilities
    - p2: Power factor for the distributions of the digits
    - hospital_specialization: String used as a seed to influence probabilities for K-V + Z
    - seed: Random seed for reproducibility

    Returns:
    - DataFrame with patient_id, hospital, and admission_code
    """
    np.random.seed(seed)

    # Sample probabilities for A-J (common to all hospitals)
    errors = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=len(LETTERS_A_J))
    probs_A_J = np.clip(0.1 + errors, 0, 1)
    probs_A_J /= probs_A_J.sum()
    probs_A_J *= p

    # Sample probabilities for K-V + Z (hospital specialization dependent)
    def get_probs_KVZ(seed_category):
        np.random.seed(hash(seed_category) % 2**32)
        rand_weights = np.random.rand(len(LETTERS_K_VZ))
        rand_probs = rand_weights / rand_weights.sum()
        rand_probs *= (1 - p)
        return rand_probs

    probs_KVZ = get_probs_KVZ(hospital_specialization)
    full_probs = np.concatenate([probs_A_J, probs_KVZ])

    # Power-law distributions for numbers
    def powerlaw_probs(max_val, power_factor):
        values = np.arange(1, max_val + 1)
        probs = values.astype(np.float64) ** (-1 / power_factor)
        return probs / probs.sum()

    probs_1_5 = powerlaw_probs(5, p2)
    probs_1_9 = powerlaw_probs(9, p2)

    # Generate mappings for categorical correlations
    clinical_specialties = [f"clinical_specialty_{l}.{i}" for l in ALL_LETTERS for i in range(1, 6)]
    billing_diagnoses = [f"billed_diagnosis_{l}.{i}" for l in ALL_LETTERS for i in range(1, 6)]
    billing_specialties = [f"billed_specialty_{i}" for i in range(20)]
    subtraject_codes = [str(i) for i in range(1, 10)]


    # Sample synthetic data
    samples = []
    for _ in range(n_samples):
        letter = np.random.choice(ALL_LETTERS, p=full_probs)
        num1 = np.random.choice(np.arange(1, 6), p=probs_1_5)
        num2 = np.random.choice(np.arange(1, 10), p=probs_1_9)
        admission_code = f"{letter}.{num1}.{num2}"

        # Feature generation
        specialty = (
            f"Specialty_{letter}.{num1}" if np.random.rand() < corr else np.random.choice(clinical_specialties)
        )
        diagnosis = (
            f"Diag_{letter}.{num1}" if np.random.rand() < corr2 else np.random.choice(billing_diagnoses)
        )
        billing_spec = (
            f"Billing_{hash(diagnosis) % 20}" if np.random.rand() < corr
            else f"Billing_{hash(specialty) % 20}" if np.random.rand() < corr2
            else np.random.choice(billing_specialties)
        )
        subtraject = (
            str(num2) if np.random.rand() < corr2 else np.random.choice(subtraject_codes)
        )

        # Numerical feature with low correlation
        base_age = 30 + ord(letter) % 10 * 3 + num1 + np.random.normal(0, 10)
        age = int(np.clip(base_age, 0, 100))

        # Gender
        gender = np.random.choice(["Male", "Female"])

        samples.append({
            'hospital': hospital_specialization,
            'admission_code': admission_code,
            'gender': gender,
            'clinical_specialty': specialty,
            'billing_diagnosis_code': diagnosis,
            'billing_specialty_code': billing_spec,
            'subtraject_code': subtraject,
            'age': age
        })
    return pl.DataFrame(samples)


def generate_federated_sources(
    hospital_specializations=['cardiology', 'neurology', 'oncology', 'academic', 'general'],
    samples_per_hospital=50000,
    p=0.6,
    variance=0.025,
    p2=0.5,
    corr=0.89,
    corr2=0.78,
    seed=42
):

    dfs = []  
    for hosp in hospital_specializations:
        df = generate_synthetic_admissions(
            n_samples=samples_per_hospital, 
            hospital_specialization=hosp,
            p=p,
            variance=variance,
            p2=p2,
            corr=corr,
            corr2=corr2,
            seed=seed
        )
        dfs.append(df)  

    return pl.concat(dfs)

def corrupt_target_label(df: pl.DataFrame, noise_rate: float = 0.2, seed: int = 42) -> pl.DataFrame:
    np.random.seed(seed)
    df_corrupted = df.clone()
    n_rows = df.height
    n_corrupt = int(n_rows * noise_rate)

    indices_to_corrupt = np.random.choice(n_rows, size=n_corrupt, replace=False)
    half = n_corrupt // 5 # 1/5th of the labels are swapped, the other 4/5th are shifted
    swap_indices = indices_to_corrupt[:half]
    cat_shift_indices = indices_to_corrupt[half:]

    # Swapped corruption
    admission_codes = df_corrupted.get_column("admission_code").to_numpy()
    shuffled = admission_codes[swap_indices].copy()
    np.random.shuffle(shuffled)
    for i, idx in enumerate(swap_indices):
        df_corrupted[idx.item(), "admission_code"] = shuffled[i]

    # Categorical shift corruption
    for idx in cat_shift_indices:
        idx = int(idx)  
        code = df_corrupted[idx, "admission_code"]
        parts = code.split(".")
        if len(parts) == 3:
            letter = parts[0]
            letter_idx = ord(letter)
            new_letter = chr(np.clip(letter_idx + np.random.choice([-1, 1]), ord("A"), ord("Z")))

            if np.random.rand() < 0.5:
                num2 = str(np.clip(int(parts[2]) + np.random.choice([-1, 1]), 1, 9))
                new_code = f"{letter}.{parts[1]}.{num2}"
            else:
                num1 = str(np.clip(int(parts[1]) + np.random.choice([-1, 1]), 1, 5))
                new_code = f"{new_letter}.{num1}.{parts[2]}"

            df_corrupted[idx, "admission_code"] = new_code

    return df_corrupted


if __name__ == "__main__":
    federated_data_sources = generate_federated_sources(
        hospital_specializations=['cardiology', 'neurology', 'oncology', 'academic', 'general'],
        samples_per_hospital=50000,
        p=0.6,
        variance=0.025,
        p2=0.5,
        seed=42,
        corr=0.89,
        corr2=0.78
    )
    federated_data_sources.write_csv("synthetic_admission_data.csv")