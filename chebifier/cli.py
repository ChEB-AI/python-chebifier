import click

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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output",
)
def predict(
    ensemble_config,
    smiles,
    smiles_file,
    output,
    ensemble_type,
    use_confidence,
    resolve_inconsistencies=True,
    verbose=False,
):
    """Predict ChEBI classes for SMILES strings using an ensemble model."""

    # Instantiate ensemble model
    ensemble = ENSEMBLES[ensemble_type](
        ensemble_config,
        resolve_inconsistencies=resolve_inconsistencies,
        verbose_output=verbose,
        use_confidence=use_confidence,
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
    predictions = ensemble.predict_smiles_list(smiles_list)

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
