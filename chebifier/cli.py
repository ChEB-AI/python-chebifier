import importlib.resources

import click
import yaml

from chebifier.model_registry import ENSEMBLES


@click.group()
def cli():
    """Command line interface for Chebifier."""
    pass


@cli.command()
@click.option(
    "--ensemble-config",
    "-e",
    type=click.Path(exists=True),
    default=None,
    help="Configuration file for ensemble models",
)
@click.option("--smiles", "-s", multiple=True, help="SMILES strings to predict")
@click.option(
    "--smiles-file",
    "-f",
    type=click.Path(exists=True),
    help="File containing SMILES strings (one per line)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file to save predictions (optional)",
)
@click.option(
    "--ensemble-type",
    "-t",
    type=click.Choice(ENSEMBLES.keys()),
    default="wmv-f1",
    help="Type of ensemble to use (default: Weighted Majority Voting)",
)
@click.option(
    "--chebi-version",
    "-v",
    type=int,
    default=241,
    help="ChEBI version to use for checking consistency (default: 241)",
)
@click.option(
    "--use-confidence",
    "-c",
    is_flag=True,
    default=True,
    help="Weight predictions based on how 'confident' a model is in its prediction (default: True)",
)
@click.option(
    "--resolve-inconsistencies",
    "-r",
    is_flag=True,
    default=True,
    help="Resolve inconsistencies in predictions automatically (default: True)",
)
def predict(
    ensemble_config,
    smiles,
    smiles_file,
    output,
    ensemble_type,
    chebi_version,
    use_confidence,
    resolve_inconsistencies=True,
):
    """Predict ChEBI classes for SMILES strings using an ensemble model."""
    # Load configuration from YAML file
    if not ensemble_config:
        print("Using default ensemble configuration")
        with (
            importlib.resources.files("chebifier")
            .joinpath("ensemble.yml")
            .open("r") as f
        ):
            config = yaml.safe_load(f)
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

    new_config = {}
    for model_name, entry in config.items():
        if "load_model" in entry:
            if entry["load_model"] not in model_registry:
                raise ValueError(
                    f"Model {entry['load_model']} not found in model registry. "
                    f"Available models are: {','.join(model_registry.keys())}."
                )
            new_config[model_name] = {**model_registry[entry["load_model"]], **entry}
        else:
            new_config[model_name] = entry
    config = new_config

    # Instantiate ensemble model
    ensemble = ENSEMBLES[ensemble_type](
        config,
        chebi_version=chebi_version,
        resolve_inconsistencies=resolve_inconsistencies,
    )

    # Collect SMILES strings from arguments and/or file
    smiles_list = list(smiles)
    if smiles_file:
        with open(smiles_file, "r") as f:
            smiles_list.extend([line.strip() for line in f if line.strip()])

    if not smiles_list:
        click.echo("No SMILES strings provided. Use --smiles or --smiles-file options.")
        return

    # Make predictions
    predictions = ensemble.predict_smiles_list(
        smiles_list, use_confidence=use_confidence
    )

    if output:
        # save as json
        import json

        with open(output, "w") as f:
            json.dump(
                {smiles: pred for smiles, pred in zip(smiles_list, predictions)},
                f,
                indent=2,
            )

    else:
        # Print results
        for i, (smiles, prediction) in enumerate(zip(smiles_list, predictions)):
            click.echo(f"Result for: {smiles}")
            if prediction:
                click.echo(f"  Predicted classes: {', '.join(map(str, prediction))}")
            else:
                click.echo("  No predictions")


if __name__ == "__main__":
    cli()
