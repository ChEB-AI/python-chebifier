import subprocess
import sys


def get_current_environment() -> str:
    """
    Return the path of the Python executable for the current environment.
    """
    return sys.executable


def check_package_installed(package_name: str) -> None:
    """
    Check if the given package is installed in the current Python environment.
    """
    python_exec = get_current_environment()
    try:
        subprocess.check_output(
            [python_exec, "-m", "pip", "show", package_name], stderr=subprocess.DEVNULL
        )
        print(f"✅ Package '{package_name}' is already installed.")
    except subprocess.CalledProcessError:
        raise (
            f"❌ Please install '{package_name}' into your environment: {python_exec}"
        )


if __name__ == "__main__":
    print(f"🔍 Using Python executable: {get_current_environment()}")
    check_package_installed("numpy")  # Replace with your desired package
