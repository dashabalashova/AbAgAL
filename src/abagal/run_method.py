import os
from pathlib import Path
import time
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from abagal.data.prepare_ab_ag_dataset import prepare_ab_ag_dataset
from abagal.model.methods import AbAgConvArgs, gradient
from abagal.model.qbc import query_by_committee_iter, random_iter


def run_experiment(
    method: str = 'random',
    random_seed: int = 0,
    error_rate: float = 0.05,
    result_folder: Path | str = Path(__file__).resolve().parent,
    data_path: Path | str = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'ab_ag_binding.tsv',
    base_antigens_count: int = 5,
    exp_cnt: int = 1,
    sample_frac: float = 0.1,
    results_dir: Path | str = Path(__file__).resolve().parents[2] / 'results',
):
    # Convert to Path
    result_folder = Path(result_folder)
    data_path = Path(data_path)
    results_dir = Path(results_dir)

    # Ensure output directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df_binding = prepare_ab_ag_dataset(data_path)
    df_train = prepare_ab_ag_dataset(df_binding, random_seed=random_seed, error_rate=error_rate)
    df_train = df_final.sample(frac=0.5, random_state=random_seed)
    df_train['AbSeq'] = df_train.AbSlideSeq
    df_train['AASeq'] = df_train.apply(lambda row: row.AgSeq + row.AbSeq, axis=1)

    conditions = [
        (df_train['AbTrain']  &  df_train['AgTrain']),      # both True
        (df_train['AbTrain']  & ~df_train['AgTrain']),      # AbTrain=True,  AgTrain=False
        (~df_train['AbTrain'] &  df_train['AgTrain']),      # AbTrain=False, AgTrain=True
        (~df_train['AbTrain'] & ~df_train['AgTrain']),      # both False
    ]
    choices = ['train','testAG','testAB','test',]
    df_train['total_split'] = np.select(conditions, choices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterations = df_train[df_train.total_split=='train'].AgSeq.drop_duplicates().shape[0] - base_antigens_count
    print(f"Experiment: {method}, random_state: {random_state}")

    if method == 'random':
        df_res = random_iter(
            dataset=df_train,
            committee_size=1,
            iterations=iterations,
            base_antigens_count=base_antigens_count,
            training_args=AbAgConvArgs(),
            device=device,
            random_state=random_seed,
        )
    elif method == 'qbc':
        df_res = query_by_committee_iter(
            dataset=df_train,
            committee_size=5,
            iterations=iterations,
            base_antigens_count=base_antigens_count,
            training_args=AbAgConvArgs(),
            device=device,
            random_state=random_seed,
            dis_quantile=error_rate,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    df_res['ags_number'] = df_res.iter + base_antigens_count
    result_path = results_dir / f'roc_aucs_{method}_{random_seed}.tsv'
    df_res.to_csv(result_path, sep='\t', index=False)
    print(f"Results saved to {result_path}")

    return None

run_experiment()