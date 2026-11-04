from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from propagator.cli.console import setup_console
from propagator.ff import ForeFireScriptRunner


class ForeFireScriptCLI(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    script: Path = Field(..., description="Path to a .ff script file")
    core: Literal["numba", "rust"] = Field(
        "numba",
        description="Simulation core: 'numba' (default) or 'rust'.",
    )
    seed: Optional[int] = Field(
        None, description="Seed for the simulation RNGs"
    )
    freeze_dir: Optional[Path] = Field(
        None,
        description="Directory for freezing burned-out tiles to disk",
    )
    verbose: CliImplicitFlag[bool] = Field(
        False, description="Enable verbose output"
    )

    def _check_script_file(self) -> None:
        if not self.script.is_file():
            raise ValueError(f"Script file not found: {self.script}")


def main() -> None:
    cli = ForeFireScriptCLI()  # type: ignore
    cli._check_script_file()

    setup_console()

    runner = ForeFireScriptRunner(
        core=cli.core,
        seed=cli.seed,
        freeze_dir=str(cli.freeze_dir) if cli.freeze_dir else None,
        verbose=cli.verbose,
        case_directory=str(cli.script.parent),
    )
    runner.run_file(cli.script)


if __name__ == "__main__":
    main()
