"""Run the Enoki command-line interface with ``python -m enoki``."""


def main() -> None:
    from enoki_cli import app

    app()


if __name__ == "__main__":
    main()
