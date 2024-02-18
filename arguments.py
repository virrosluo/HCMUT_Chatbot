import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no_reindex", help="Reindex all database", action="store_false", default=False
)
parser.add_argument(
    "--dev", help="Enable developing mode", action="store_true", default=False
)
args = parser.parse_args()
