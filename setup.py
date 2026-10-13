import platform
import sys

from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def _get_platform_tag():
    if platform.system() != "Linux":
        return platform.system().lower()

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "manylinux1_x86_64"
    if machine in ("aarch64", "arm64"):
        return "manylinux2014_aarch64"
    return f"linux_{machine}"


# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"
        platform_tag = _get_platform_tag()

        return python_version, abi_tag, platform_tag


# Setup configuration
setup(
    version="0.1.0",
    packages=find_packages(include=["miles*", "miles_plugins*"]),
    include_package_data=True,
    package_data={"miles.dashboard": ["static/*"]},
    cmdclass={"bdist_wheel": bdist_wheel},
)
