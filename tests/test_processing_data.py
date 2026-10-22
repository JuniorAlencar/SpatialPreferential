import sys
from pathlib import Path

import pandas as pd


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from src.processing_data import processing_all_data


def test_processing_writes_aggregate_into_data_root(tmp_path):
    prop_dir = (
        tmp_path / "Data" / "N_100" / "m0_2" / "dim_3"
        / "alpha_a_1.50_alpha_g_2.00" / "prop"
    )
    prop_dir.mkdir(parents=True)

    header = (
        "#MeanShortestPathDijstrika,MeanShortestPathBFS,Ass_Spearman,"
        "Ass_Spearman_error,Ass_Newman,Ass_Newman_error,ClusterCoefficient\n"
    )
    (prop_dir / "prop_1.csv").write_text(
        header + "2,4,0.1,0.01,0.2,0.02,0.3\n"
    )
    (prop_dir / "prop_2.csv").write_text(
        header + "4,6,0.3,0.03,0.4,0.04,0.5\n"
    )

    _, result = processing_all_data(tmp_path / "Data")

    output_path = tmp_path / "Data" / "all_data.csv"
    assert output_path.exists()
    persisted = pd.read_csv(output_path)
    assert len(result) == len(persisted) == 1

    row = persisted.iloc[0]
    assert (row["N"], row["m0"], row["dim"]) == (100, 2, 3)
    assert (row["alpha_a"], row["alpha_g"]) == (1.5, 2.0)
    assert row["N_samples"] == 2
    assert row["Short_Cost_mean"] == 3.0
    assert row["Short_BFS_mean"] == 5.0
