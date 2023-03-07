"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """ML-Ekosystem."""


if __name__ == "__main__":
    main(prog_name="mleko")  # pragma: no cover
