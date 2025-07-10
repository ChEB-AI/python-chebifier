


import click
import yaml
from chebifier.ensemble.base_ensemble import BaseEnsemble
from chebifier.ensemble.weighted_majority_ensemble import WMVwithPPVNPVEnsemble, WMVwithF1Ensemble


@click.group()
def cli():
    """Command line interface for Chebifier."""
    pass

ENSEMBLES = {
    "mv": BaseEnsemble,
    "wmv-ppvnpv": WMVwithPPVNPVEnsemble,
    "wmv-f1": WMVwithF1Ensemble
}

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--smiles', '-s', multiple=True, help='SMILES strings to predict')
@click.option('--smiles-file', '-f', type=click.Path(exists=True), help='File containing SMILES strings (one per line)')
@click.option('--output', '-o', type=click.Path(), help='Output file to save predictions (optional)')
@click.option('--ensemble-type', '-e', type=click.Choice(ENSEMBLES.keys()), default='mv', help='Type of ensemble to use (default: Majority Voting)')
@click.option("--chebi-version", "-v", type=int, default=241, help="ChEBI version to use for checking consistency (default: 241)")
@click.option("--use-confidence", "-c", is_flag=True, default=True, help="Weight predictions based on how 'confident' a model is in its prediction (default: True)")
def predict(config_file, smiles, smiles_file, output, ensemble_type, chebi_version):
    """Predict ChEBI classes for SMILES strings using an ensemble model.
    
    CONFIG_FILE is the path to a YAML configuration file for the ensemble model.
    """
    # Load configuration from YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Instantiate ensemble model
    ensemble = ENSEMBLES[ensemble_type](config, chebi_version=chebi_version)
    
    # Collect SMILES strings from arguments and/or file
    smiles_list = list(smiles)
    if smiles_file:
        with open(smiles_file, 'r') as f:
            smiles_list.extend([line.strip() for line in f if line.strip()])
    
    if not smiles_list:
        click.echo("No SMILES strings provided. Use --smiles or --smiles-file options.")
        return

    # Make predictions
    predictions = ensemble.predict_smiles_list(smiles_list)

    if output:
        # save as json
        import json
        with open(output, 'w') as f:
            json.dump({smiles: pred for smiles, pred in zip(smiles_list, predictions)}, f, indent=2)

    else:
        # Print results
        for i, (smiles, prediction) in enumerate(zip(smiles_list, predictions)):
            click.echo(f"Result for: {smiles}")
            if prediction:
                click.echo(f"  Predicted classes: {', '.join(map(str, prediction))}")
            else:
                click.echo("  No predictions")


if __name__ == '__main__':
    cli()
