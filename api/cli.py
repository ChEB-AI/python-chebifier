from pathlib import Path

import click
import yaml

from chebifier.model_registry import ENSEMBLES, MODEL_TYPES

from .check_env import check_package_installed, get_current_environment
from .hugging_face import download_model_files

yaml_path = Path("api/api_registry.yml")
if yaml_path.exists():
    with yaml_path.open("r") as f:
        api_registry = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"{yaml_path} not found.")


@click.group()
def cli():
    """Command line interface for Api-Chebifier."""
    pass


@cli.command()
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
    "--model-type",
    "-m",
    type=click.Choice(api_registry.keys()),
    default="mv",
    help="Type of model to use",
)
def predict(smiles, smiles_file, output, model_type):
    """Predict ChEBI classes for SMILES strings using an ensemble model.

    CONFIG_FILE is the path to a YAML configuration file for the ensemble model.
    """

    # Collect SMILES strings from arguments and/or file
    smiles_list = list(smiles)
    if smiles_file:
        with open(smiles_file, "r") as f:
            smiles_list.extend([line.strip() for line in f if line.strip()])

    if not smiles_list:
        click.echo("No SMILES strings provided. Use --smiles or --smiles-file options.")
        return

    print("Current working environment is:", get_current_environment())

    def get_individual_model(model_config):
        predictor_kwargs = {}
        if "hugging_face" in model_config:
            predictor_kwargs = download_model_files(model_config["hugging_face"])
        check_package_installed(model_config["package_name"])
        return predictor_kwargs

    if model_type in MODEL_TYPES:
        print(f"Predictor for Single/Individual Model: {model_type}")
        model_config = api_registry[model_type]
        predictor_kwargs = get_individual_model(model_config)
        predictor_kwargs["model_name"] = model_type
        model_instance = MODEL_TYPES[model_type](**predictor_kwargs)

    elif model_type in ENSEMBLES:
        print(f"Predictor for Ensemble Model: {model_type}")
        ensemble_config = {}
        for i, en_comp in enumerate(api_registry[model_type]["ensemble_of"]):
            assert en_comp in MODEL_TYPES
            print(f"For ensemble component {en_comp}")
            predictor_kwargs = get_individual_model(api_registry[en_comp])
            model_key = f"model_{i + 1}"
            ensemble_config[model_key] = {
                "type": en_comp,
                "model_name": f"{en_comp}_{model_key}",
                **predictor_kwargs,
            }
        model_instance = ENSEMBLES[model_type](ensemble_config)

    else:
        raise ValueError("")

    # Make predictions
    predictions = model_instance.predict_smiles_list(smiles_list)

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
