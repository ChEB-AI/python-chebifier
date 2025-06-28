import importlib
from pathlib import Path

import click
import yaml

from chebifier.prediction_models.base_predictor import BasePredictor

from .hugging_face import download_model_files
from .setup_env import SetupEnvAndPackage

yaml_path = Path("api/registry.yml")
if yaml_path.exists():
    with yaml_path.open("r") as f:
        model_registry = yaml.safe_load(f)
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
    type=click.Choice(model_registry.keys()),
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

    model_config = model_registry[model_type]
    predictor_kwargs = {"model_name": model_type}

    current_dir = Path(__file__).resolve().parent

    if "hugging_face" in model_config:
        local_file_path = download_model_files(
            model_config["hugging_face"],
            current_dir / ".api_models" / model_type,
        )
        predictor_kwargs["ckpt_path"] = local_file_path["ckpt"]
        predictor_kwargs["target_labels_path"] = local_file_path["labels"]

    SetupEnvAndPackage().setup(
        repo_url=model_config["repo_url"],
        clone_dir=current_dir / ".cloned_repos",
        venv_dir=current_dir,
    )

    model_cls_path = model_config["wrapper"]
    module_path, class_name = model_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls: type = getattr(module, class_name)
    model_instance = model_cls(**predictor_kwargs)
    assert isinstance(model_instance, BasePredictor)

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
