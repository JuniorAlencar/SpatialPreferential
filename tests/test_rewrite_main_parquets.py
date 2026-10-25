import gzip
import sys
from pathlib import Path

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
from rewrite_main_parquets import rewrite_parquets


def test_rewrite_parquets_converts_gml_and_writes_manifest(tmp_path):
    source_dir = tmp_path / "Data" / "N_10" / "m0_2" / "dim_3" / "alpha_a_1.50_alpha_g_2.00"
    gml_dir = source_dir / "gml"
    gml_dir.mkdir(parents=True)
    source = gml_dir / "gml_11.gml.gz"
    content = """graph
[
node
[
id 1
graphics
[
x 1.0
y 2.0
z 3.0
]
degree 2
]
edge
[
source 1
target 2
distance 0.25
]
]
"""
    source.write_bytes(gzip.compress(content.encode()))
    original = source.read_bytes()

    manifest = rewrite_parquets(tmp_path / "Data", tmp_path / "processed", workers=1)

    output = tmp_path / "processed" / "parquets"
    relative_dir = Path("N_10") / "dim_3"
    nodes = output / relative_dir / "N10_d3_m2_G2.0_A1.5_seed001to001_nodes.parquet"
    edges = output / relative_dir / "N10_d3_m2_G2.0_A1.5_seed001to001_edges.parquet"
    assert len(manifest) == 1
    assert source.read_bytes() == original
    assert pd.read_parquet(nodes).columns.tolist() == ["Id", "x", "y", "z", "deg", "seed"]
    assert pd.read_parquet(edges).columns.tolist() == ["source", "target", "len", "seed"]
    assert pd.read_csv(output / "manifest.csv")["source_path"].tolist() == [str(source.relative_to(tmp_path / "Data"))]
    assert pd.read_csv(output / "manifest.csv")["nodes_file"].tolist() == [str(relative_dir / nodes.name)]

    # A ausência do bruto não remove a amostra já processada do manifesto.
    source.unlink()
    manifest = rewrite_parquets(tmp_path / "Data", tmp_path / "processed", workers=1)
    assert len(manifest) == 1

    # Um segundo GML é anexado; o primeiro é preservado via manifesto.
    second = gml_dir / "gml_12.gml.gz"
    second.write_bytes(gzip.compress(content.encode()))
    manifest = rewrite_parquets(tmp_path / "Data", tmp_path / "processed", workers=1)
    updated_nodes = output / relative_dir / "N10_d3_m2_G2.0_A1.5_seed001to002_nodes.parquet"
    assert len(manifest) == 2
    assert not nodes.exists()
    assert len(pd.read_parquet(updated_nodes)) == 2

    # Sem GML novo, não há nova conversão nem duplicação no manifesto.
    manifest = rewrite_parquets(tmp_path / "Data", tmp_path / "processed", workers=1)
    assert len(manifest) == 2
