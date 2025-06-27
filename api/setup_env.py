import os
import re
import subprocess
import sys
from pathlib import Path

# Conditional import of tomllib based on Python version
if sys.version_info >= (3, 11):
    import tomllib  # built-in in Python 3.11+
else:
    import toml as tomllib  # use third-party toml library for older versions


class SetupEnvAndPackage:
    """Utility class for cloning a repository, setting up a virtual environment, and installing a package."""

    def setup(
        self,
        repo_url: str,
        clone_dir: Path,
        venv_dir: Path,
        venv_name: str = ".venv-chebifier",
    ) -> None:
        """
        Orchestrates the full setup process: cloning the repository,
        creating a virtual environment, and installing the package.

        Args:
            repo_url (str): URL of the Git repository.
            clone_dir (Path): Directory to clone the repo into.
            venv_dir (Path): Directory where the virtual environment will be created.
            venv_name (str): Name of the virtual environment folder.
        """
        cloned_repo_path = self._clone_repo(repo_url, clone_dir)
        venv_path = self._create_virtualenv(venv_dir, venv_name)
        self._install_from_pyproject(venv_path, cloned_repo_path)

    def _clone_repo(self, repo_url: str, clone_dir: Path) -> Path:
        """
        Clone a Git repository into a specified directory.

        Args:
            repo_url (str): Git URL to clone.
            clone_dir (Path): Directory to clone into.

        Returns:
            Path: Path to the cloned repository.
        """
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        clone_path = Path(clone_dir or repo_name)

        if not clone_path.exists():
            print(f"Cloning {repo_url} into {clone_path}...")
            subprocess.check_call(
                ["git", "clone", "--depth=1", repo_url, str(clone_path)]
            )
        else:
            print(f"Repo already exists at {clone_path}")

        return clone_path

    @staticmethod
    def _create_virtualenv(venv_dir: Path, venv_name: str = ".venv-chebifier") -> Path:
        """
        Create a virtual environment at the specified path.

        Args:
            venv_dir (Path): Base directory where the venv will be created.
            venv_name (str): Name of the virtual environment directory.

        Returns:
            Path: Path to the virtual environment.
        """
        venv_path = venv_dir / venv_name

        if venv_path.exists():
            print(f"Virtual environment already exists at: {venv_path}")
            return venv_path

        print(f"Creating virtual environment at: {venv_path}")

        try:
            subprocess.check_call(["virtualenv", str(venv_path)])
        except FileNotFoundError:
            print("virtualenv not found, installing it now...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "virtualenv"]
            )
            subprocess.check_call(["virtualenv", str(venv_path)])

        return venv_path

    def _install_from_pyproject(self, venv_dir: Path, cloned_repo_path: Path) -> None:
        """
        Install the cloned package in editable mode.

        Args:
            venv_dir (Path): Path to the virtual environment.
            cloned_repo_path (Path): Path to the cloned repository.
        """
        pip_executable = (
            venv_dir / "Scripts" / "pip.exe"
            if os.name == "nt"
            else venv_dir / "bin" / "pip"
        )

        if not pip_executable.exists():
            raise FileNotFoundError(f"pip not found at {pip_executable}")

        try:
            package_name = self._get_package_name(cloned_repo_path)
        except Exception as e:
            raise RuntimeError(f"Error extracting package name: {e}")

        try:
            subprocess.check_output(
                [str(pip_executable), "show", package_name], stderr=subprocess.DEVNULL
            )
            print(f"Package '{package_name}' is already installed.")
        except subprocess.CalledProcessError:
            print(f"Installing '{package_name}' from {cloned_repo_path}...")
            subprocess.check_call(
                [str(pip_executable), "install", "-e", "."],
                cwd=cloned_repo_path,
            )

    @staticmethod
    def _get_package_name(cloned_repo_path: Path) -> str:
        """
        Extracts the package name from `pyproject.toml` or `setup.py`.

        Args:
            cloned_repo_path (Path): Path to the cloned repository.

        Returns:
            str: Name of the Python package.

        Raises:
            ValueError: If parsing fails.
            FileNotFoundError: If neither config file is found.
        """
        pyproject_path = cloned_repo_path / "pyproject.toml"
        setup_path = cloned_repo_path / "setup.py"

        if pyproject_path.exists():
            try:
                with pyproject_path.open("rb") as f:
                    pyproject = tomllib.load(f)
                return pyproject["project"]["name"]
            except Exception as e:
                raise ValueError(f"Failed to parse pyproject.toml: {e}")

        elif setup_path.exists():
            try:
                setup_contents = setup_path.read_text()
                match = re.search(r'name\s*=\s*[\'"]([^\'"]+)[\'"]', setup_contents)
                if match:
                    return match.group(1)
                else:
                    raise ValueError("Could not find package name in setup.py")
            except Exception as e:
                raise ValueError(f"Failed to parse setup.py: {e}")

        else:
            raise FileNotFoundError("Neither pyproject.toml nor setup.py found.")
