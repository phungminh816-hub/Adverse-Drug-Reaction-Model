"""
adr – ADR prediction from molecular SMILES.

Package structure:
  adr.config       – load/merge/validate YAML configs
  adr.schemas      – dataclass contracts cho artifact và prediction
  adr.data         – download, validate, prepare, split
  adr.features     – fingerprint và character tokenizer
  adr.models       – frequency baseline, logreg, mlp, cnn
  adr.train        – training loops và losses
  adr.evaluate     – metrics và evaluation runner
  adr.predict      – predictor thống nhất và predict CLI
  adr.utils        – seed, artifact paths, helpers
"""
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

__version__ = "0.1.0"
