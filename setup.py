import re
from pathlib import Path

import setuptools
from setuptools_rust import Binding, RustExtension

# Setup requires
setup_requires = [
    "setuptools-rust>=0.11.1",
    "wheel",
]

# Dependencies
dependencies = {
    "numpy": "1.20.2",
    "matplotlib": "3.4.1",
    "scipy": "1.6.2",
    "numba": "0.53.1",
    "rich": "10.2.0",
    "requests": "2.25.1",
}

# -- Constants --
BASE_DIR = Path(__file__).resolve().parent
README = Path(BASE_DIR / "README.md").read_text(encoding="utf8")

URL = "https://github.com/Deep-Alchemy/Mini-Keras"

# -- Version config --
VERSION = re.search(
    r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    Path(BASE_DIR / "mini_keras/__init__.py").read_text(),
    re.MULTILINE,
).group(1)

if not VERSION:
    raise RuntimeError("VERSION is not set!")

# -- Setup --
setuptools.setup(
    # Project info
    name="Mini-Keras",
    version=VERSION,

    author="Deep Alchemy Team",
    author_email="warriordefenderz@gmail.com",

    description="An advanced and lightweight ML and deep learning library for python.",
    long_description=README,
    long_description_content_type="text/markdown",

    # Project repo info
    license="MIT",
    url=URL,
    project_urls={"Documentation": URL, "Issue tracker": f"{URL}/issues"},

    # Packages in the project
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "tools", "tools.*"]),

    # Dependencies for the package
    install_requires=[f"{k}=={v}" for k, v in dependencies.items()],

    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],

    # Python minimum version
    python_requires=">=3.7",

    # Rust extension config
    rust_extensions=[
        RustExtension(
            "mini_keras._mini_keras",
            binding=Binding.PyO3
        ),
    ],
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False,
)
