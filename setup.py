import re
from pathlib import Path

import setuptools

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
README = Path(BASE_DIR / "README.md").read_text()

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
    name="Mini-Keras",
    version=VERSION,
    author="Deep Alchemy Team",
    author_email="warriordefenderz@gmail.com",
    description="An advanced and lightweight ML and deep learning library for python.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="GPL v3",
    url=URL,
    project_urls={"Documentation": URL, "Issue tracker": f"{URL}/issues"},
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "tools", "tools.*"]),
    install_requires=[f"{k}=={v}" for k, v in dependencies.items()],
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
    python_requires=">=3.7",
)