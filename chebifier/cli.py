import importlib.resources
import os
from typing import Literal

import click
import pandas as pd
import yaml
from chebi_utils.read_molecule import smiles_or_inchi_to_mol

from chebifier.build_ensemble import EnsembleBuilder
from chebifier.check_env import check_package_installed
from chebifier.hugging_face import download_model_files
from chebifier.model_registry import ENSEMBLES, MODEL_TYPES
from chebifier.predict import predict as predict_molecules
from chebifier.utils import get_default_configs, load_chebi_graph, process_config


def read_molecules(molecules, molecule_file):
    """Collect molecules from CLI arguments and/or a file (one molecule per line) and convert them to RDKit mol objects."""
    raw_inputs = list(molecules)
    if molecule_file:
        with open(molecule_file, "r", encoding="utf-8") as f:
            raw_inputs.extend([line.strip() for line in f if line.strip()])

    return [smiles_or_inchi_to_mol(raw_input) for raw_input in raw_inputs]


def build_base_learners(ensemble_config):
    """Instantiate the base learners described by an ensemble configuration file."""
    if ensemble_config is None:
        config = get_default_configs()
    else:
        print(f"Loading ensemble configuration from {ensemble_config}")
        with open(ensemble_config, "r") as f:
            config = yaml.safe_load(f)

    with (
        importlib.resources.files("chebifier")
        .joinpath("model_registry.yml")
        .open("r") as f
    ):
        model_registry = yaml.safe_load(f)

    chebi_graph = load_chebi_graph()
    base_learners = {}
    for model_name, model_config in process_config(config, model_registry).items():
        if "hugging_face" in model_config:
            hugging_face_kwargs = download_model_files(model_config["hugging_face"])
        else:
            hugging_face_kwargs = {}
        if "package_name" in model_config:
            check_package_installed(model_config["package_name"])
        base_learners[model_name] = MODEL_TYPES[model_config["type"]](
            model_name,
            **model_config,
            **hugging_face_kwargs,
            chebi_graph=chebi_graph,
        )
    return base_learners


def load_dataset(data_path, split: Literal["train", "validation", "test"]):
    data_file = os.path.join(data_path, "data.pkl")
    splits_file = os.path.join(data_path, "splits.csv")
    if not os.path.exists(data_file) or not os.path.exists(splits_file):
        raise FileNotFoundError(
            f"Required dataset files not found. Expected to find 'data.pkl' and 'splits.csv' in the provided data path ({data_path})."
        )
    data_df = pd.read_pickle(data_file)
    splits_df = pd.read_csv(splits_file)
    # merge dataframe on id column and filter by the specified split
    splits_df["id"] = splits_df["id"].astype(str)
    merged_df = data_df.merge(splits_df, left_on="chebi_id", right_on="id", how="inner")
    merged_df = merged_df[merged_df["split"] == split].reset_index(drop=True)

    mol_list = merged_df["mol"].tolist()

    # extract labels from data_df: every column other than chebi_id/mol/id/split is a ChEBI class label
    label_columns = [c for c in data_df.columns if c not in ("chebi_id", "mol")]
    labels_df = merged_df[label_columns].astype(bool)
    labels_df.columns = [str(c) for c in label_columns]

    print(
        f"Loaded {len(mol_list)} molecules and {len(labels_df.columns)} labels for split '{split}' from {data_path}."
    )

    return mol_list, labels_df


def ensemble_options(command):
    """Options shared by all commands that use an ensemble."""
    for option in reversed(
        [
            click.option(
                "--ensemble-config",
                "-e",
                type=click.Path(exists=True),
                default=None,
                help="Configuration file listing the base learners of the ensemble",
            ),
            click.option(
                "--ensemble-type",
                "-t",
                type=click.Choice(ENSEMBLES.keys()),
                default="wmv-f1",
                help="Type of ensemble to use (default: Weighted Majority Voting with F1 weights)",
            ),
            click.option(
                "--ensemble-dir",
                "-d",
                type=click.Path(),
                required=True,
                help="Directory where the calibration results of the ensemble are stored",
            ),
            click.option(
                "--prediction-cache-dir",
                type=click.Path(),
                default=None,
                help="Directory for caching base learner predictions",
            ),
        ]
    ):
        command = option(command)
    return command


def data_options(command):
    """Options shared by the commands that work on a ChEBI dataset split."""
    for option in reversed(
        [
            click.option(
                "--data-path",
                type=str,
                required=True,
                help="Data source: local dataset directory or Hugging Face repo id",
            ),
        ]
    ):
        command = option(command)
    return command


@click.group()
def cli():
    """Command line interface for Chebifier."""
    pass


@cli.command()
@ensemble_options
@data_options
def build(
    ensemble_config, ensemble_type, ensemble_dir, prediction_cache_dir, data_path
):
    """Build (calibrate) an ensemble on the ChEBI validation set."""
    base_learners = build_base_learners(ensemble_config)
    ensemble_model = ENSEMBLES[ensemble_type](ensemble_dir)

    # TODO: Hugging Face support
    validation_data, validation_labels = load_dataset(data_path, split="validation")

    builder = EnsembleBuilder(
        base_learners,
        ensemble_model,
        validation_data,
        validation_labels,
        prediction_cache_dir,
    )
    builder.build_ensemble()


@cli.command()
@ensemble_options
@data_options
@click.option(
    "--resolve-inconsistencies/--no-resolve-inconsistencies",
    default=True,
    help="Resolve inconsistencies in the aggregated predictions (default: True)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file to save the evaluation results (optional)",
)
def evaluate(
    ensemble_config,
    ensemble_type,
    ensemble_dir,
    prediction_cache_dir,
    data_path,
    resolve_inconsistencies,
    output,
):
    """Evaluate an ensemble on the ChEBI test set."""
    base_learners = build_base_learners(ensemble_config)
    ensemble_model = ENSEMBLES[ensemble_type](ensemble_dir)

    # TODO: Hugging Face support
    test_data, test_labels = load_dataset(data_path, split="test")

    predictions = predict_molecules(
        base_learners,
        ensemble_model,
        test_data,
        prediction_cache_dir=prediction_cache_dir,
        resolve_inconsistencies=resolve_inconsistencies,
    )

    print(f"Predictions: {predictions}")

    # TODO: compare predictions to test_labels, report metrics and save them to output


@cli.command()
@ensemble_options
@click.option(
    "--molecules", "-m", multiple=True, help="SMILES or InChI strings to predict"
)
@click.option(
    "--molecule-file",
    "-f",
    type=click.Path(exists=True),
    default=None,
    help="File containing SMILES or InChI strings (one per line)",
)
@click.option(
    "--resolve-inconsistencies/--no-resolve-inconsistencies",
    default=True,
    help="Resolve inconsistencies in the aggregated predictions (default: True)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file to save the predictions (optional)",
)
@click.option(
    "--decision-threshold",
    "-dt",
    type=float,
    default=0,
    help="Threshold for classifying predictions (default: 0)",
)
def predict(
    ensemble_config,
    ensemble_type,
    ensemble_dir,
    prediction_cache_dir,
    molecules,
    molecule_file,
    resolve_inconsistencies,
    decision_threshold,
    output,
):
    """Predict ChEBI classes for a list of SMILES / InChI strings."""
    molecules_list = read_molecules(molecules, molecule_file)
    if not molecules_list:
        click.echo("No molecules provided. Use --molecules or --molecule-file.")
        return

    base_learners = build_base_learners(ensemble_config)
    ensemble_model = ENSEMBLES[ensemble_type](ensemble_dir)

    predictions = predict_molecules(
        base_learners,
        ensemble_model,
        molecules_list,
        prediction_cache_dir=prediction_cache_dir,
        resolve_inconsistencies=resolve_inconsistencies,
        decision_threshold=decision_threshold,
    )

    print(f"Predictions: {predictions}")
    # TODO: turn the aggregated predictions into ChEBI classes per molecule, print them / save to output


if __name__ == "__main__":
    cli()
