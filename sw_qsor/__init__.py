"""sw-qsor: predicting what a molecule smells like, and testing such predictions honestly.

The package never reads a file its caller did not name. Data is fetched by the scripts in
``scripts/`` into a directory the caller chooses, and loaders take that path explicitly.
"""

__version__ = "0.0.1"
