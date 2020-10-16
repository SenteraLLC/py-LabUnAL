"""CLI entrypoints through ``click`` bindings."""

import logging

import click

import py-LabUnAL_pkg


@click.group()
@click.option(
    "--log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="INFO",
    help="Set logging level for both console and file",
)
def cli(log_level):
    """CLI entrypoint."""
    logging.basicConfig(level=log_level)


@cli.command()
def version():
    """Print application version."""
    print(f"py-LabUnAL_pkg version\t{py-LabUnAL_pkg.__version__}")
