import importlib.resources
import os
from typing import Literal

import click
import numpy as np
import pandas as pd
import yaml
from chebi_utils.read_molecule import smiles_or_inchi_to_mol

from chebifier.build_ensemble import EnsembleBuilder
from chebifier.check_env import check_package_installed
from chebifier.hugging_face import download_model_files
from chebifier.inconsistency_resolution import SMOOTHER_NAMES
from chebifier.model_registry import ENSEMBLES, MODEL_TYPES
from chebifier.predict import aggregate_predictions, base_learner_cache_path
from chebifier.predict import predict as predict_molecules
from chebifier.predict import resolve_and_decide
from chebifier.utils import (
    get_default_configs,
    get_disjoint_files,
    load_chebi_graph,
    process_config,
)


def read_molecules(molecules, molecule_file):
    """Collect molecules from CLI arguments and/or a file (one molecule per line) and convert them to RDKit mol objects."""
    raw_inputs = list(molecules)
    if molecule_file:
        with open(molecule_file, "r", encoding="utf-8") as f:
            raw_inputs.extend([line.strip() for line in f if line.strip()])

    return [smiles_or_inchi_to_mol(raw_input) for raw_input in raw_inputs]


def build_base_learners(ensemble_config, prediction_cache_dir=None, split=None):
    """Instantiate the base learners described by an ensemble configuration file.

    If prediction_cache_dir and split are given, models whose predictions for that split are
    already cached are not instantiated (their entry is None) - loading their checkpoints would
    be a waste of time since the cached predictions are used instead.
    """
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

    chebi_graph = None
    base_learners = {}
    for model_name, model_config in process_config(config, model_registry).items():
        if (
            prediction_cache_dir is not None
            and split is not None
            and os.path.exists(
                base_learner_cache_path(prediction_cache_dir, model_name, split)
            )
        ):
            print(f"{model_name} {split} predictions found in cache, skipping model.")
            base_learners[model_name] = None
            continue
        if chebi_graph is None:
            chebi_graph = load_chebi_graph()
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


def parse_ir_params(ir_param):
    params = {}
    for entry in ir_param:
        if "=" not in entry:
            raise click.BadParameter(f"Expected key=value, got '{entry}'")
        key, value = entry.split("=", 1)
        try:
            params[key.strip()] = float(value)
        except ValueError:
            params[key.strip()] = value
    return params


def parse_ensemble_params(ensemble_param):
    """Parse key=value arguments into keyword arguments for the ensemble constructor. Unlike the
    inconsistency resolution parameters, these are coerced to int where possible - passing a float
    where the ensemble expects a count (e.g. region_size) fails deep inside numpy."""
    params = {}
    for entry in ensemble_param:
        if "=" not in entry:
            raise click.BadParameter(f"Expected key=value, got '{entry}'")
        key, value = entry.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        params[key.strip()] = value
    return params


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


def base_learner_options(command):
    """Options shared by all commands that use base learners."""
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
                "--prediction-cache-dir",
                type=click.Path(),
                default=None,
                help="Directory for caching base learner predictions",
            ),
        ]
    ):
        command = option(command)
    return command


def ensemble_options(command):
    """Options shared by all commands that use a single ensemble."""
    for option in reversed(
        [
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
        ]
    ):
        command = option(command)
    return base_learner_options(command)


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
@click.option(
    "--ensemble-param",
    "-ep",
    multiple=True,
    help="Extra key=value argument for the ensemble constructor, e.g. -ep candidate_k=70 "
    "(repeatable). Parameters that change the stored model are written to the ensemble's "
    "metadata, so 'evaluate' does not need to be given them again.",
)
def build(
    ensemble_config,
    ensemble_type,
    ensemble_dir,
    prediction_cache_dir,
    data_path,
    ensemble_param,
):
    """Build (calibrate) an ensemble on the ChEBI validation set."""
    base_learners = build_base_learners(ensemble_config)
    ensemble_model = ENSEMBLES[ensemble_type](
        ensemble_dir, **parse_ensemble_params(ensemble_param)
    )

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
@base_learner_options
@data_options
@click.option(
    "--ensemble-type",
    "-t",
    type=click.Choice(ENSEMBLES.keys()),
    multiple=True,
    default=("wmv-f1",),
    help="Type of ensemble to evaluate (repeatable, paired with --ensemble-dir)",
)
@click.option(
    "--ensemble-dir",
    "-d",
    type=click.Path(),
    multiple=True,
    required=True,
    help="Directory where the calibration results of the ensemble are stored (one per --ensemble-type)",
)
@click.option(
    "--resolve-inconsistencies/--no-resolve-inconsistencies",
    default=True,
    help="Resolve inconsistencies in the aggregated predictions (default: True). "
    "--no-resolve-inconsistencies is equivalent to '-ir none'.",
)
@click.option(
    "--inconsistency-resolution",
    "-ir",
    type=click.Choice(SMOOTHER_NAMES + ["none"]),
    multiple=True,
    default=("score-based",),
    help="Method used to resolve inconsistencies (repeatable, default: score-based). "
    "All methods share the base learner predictions and the ensemble aggregation.",
)
@click.option(
    "--ir-param",
    "-irp",
    multiple=True,
    help="Extra key=value parameter for the resolution method, e.g. -irp k=2.0 (repeatable)",
)
@click.option(
    "--split",
    type=click.Choice(["validation", "test"]),
    default="test",
    help="Dataset split to evaluate on (default: test)",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip ensemble / resolution combinations whose output file already exists",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file for the ensemble predictions, only allowed for a single ensemble and "
    "resolution method (default: <ensemble-dir>/<split>_predictions_<method>.npz, "
    "with 'noir' as method for '-ir none')",
)
def evaluate(
    ensemble_config,
    ensemble_type,
    ensemble_dir,
    prediction_cache_dir,
    data_path,
    resolve_inconsistencies,
    inconsistency_resolution,
    ir_param,
    split,
    skip_existing,
    output,
):
    """Store the predictions of one or more ensembles on a ChEBI dataset split."""
    if len(ensemble_type) != len(ensemble_dir):
        raise click.BadParameter(
            f"Got {len(ensemble_type)} --ensemble-type and {len(ensemble_dir)} --ensemble-dir "
            f"values, expected one directory per ensemble type."
        )
    variants = list(inconsistency_resolution) if resolve_inconsistencies else ["none"]
    if output is not None and (len(ensemble_type) > 1 or len(variants) > 1):
        raise click.BadParameter(
            "--output can only be used with a single --ensemble-type and a single "
            "--inconsistency-resolution."
        )

    def output_path(dir_, variant):
        if output is not None:
            return output
        suffix = "noir" if variant == "none" else variant
        return os.path.join(dir_, f"{split}_predictions_{suffix}.npz")

    jobs = {}
    for type_, dir_ in zip(ensemble_type, ensemble_dir):
        todo = [
            variant
            for variant in variants
            if not (skip_existing and os.path.exists(output_path(dir_, variant)))
        ]
        if todo:
            jobs[(type_, dir_)] = todo
        else:
            print(f"All outputs for {type_} in {dir_} exist, skipping.")
    if not jobs:
        return

    base_learners = build_base_learners(
        ensemble_config, prediction_cache_dir=prediction_cache_dir, split=split
    )

    # TODO: Hugging Face support
    eval_data, eval_labels = load_dataset(data_path, split=split)

    chebi_graph, disjoint_files = None, None
    if any(variant != "none" for variant in variants):
        chebi_graph = load_chebi_graph()
        disjoint_files = get_disjoint_files()

    ir_params = parse_ir_params(ir_param)
    for (type_, dir_), todo in jobs.items():
        ensemble_model = ENSEMBLES[type_](dir_)
        aggregated, predicted_classes = aggregate_predictions(
            base_learners,
            ensemble_model,
            eval_data,
            prediction_cache_dir=prediction_cache_dir,
            classes=[str(cls) for cls in eval_labels.columns],
            split=split,
        )
        for variant in todo:
            print(f"Resolving inconsistencies for {type_} with '{variant}'...")
            predictions = resolve_and_decide(
                aggregated,
                predicted_classes,
                inconsistency_resolution=variant,
                inconsistency_resolution_params=ir_params,
                chebi_graph=chebi_graph,
                disjoint_files=disjoint_files,
                decision_threshold=ensemble_model.decision_threshold,
            )
            target = output_path(dir_, variant)
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            np.savez_compressed(
                target,
                classes=np.array(predictions["predicted_classes"]),
                scores=predictions["net_score"].numpy(),
                decisions=predictions["class_decisions"].numpy(),
                has_valid_predictions=predictions["has_valid_predictions"].numpy(),
                decision_threshold=np.array(ensemble_model.decision_threshold),
            )
            print(
                f"Saved {type_} predictions for split '{split}' "
                f"({predictions['class_decisions'].shape[0]} molecules, {len(predictions['predicted_classes'])} classes, "
                f"{int(predictions['complete_failure'].sum())} molecules without any valid prediction) to {target}."
            )


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
    "--inconsistency-resolution",
    "-ir",
    type=click.Choice(SMOOTHER_NAMES),
    default="score-based",
    help="Method used to resolve inconsistencies (default: score-based)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file to save the predictions (optional)",
)
@click.option(
    "--ir-param",
    "-irp",
    multiple=True,
    help="Extra key=value parameter for the resolution method, e.g. -irp k=2.0 (repeatable)",
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
    inconsistency_resolution,
    ir_param,
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
        inconsistency_resolution=inconsistency_resolution,
        inconsistency_resolution_params=parse_ir_params(ir_param),
        decision_threshold=decision_threshold,
    )

    print(f"Predictions: {predictions}")
    # TODO: turn the aggregated predictions into ChEBI classes per molecule, print them / save to output


if __name__ == "__main__":
    cli()
