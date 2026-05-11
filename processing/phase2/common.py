from pathlib import Path

from processing.phase1.common import DATA_ROOT, PROJECT_ROOT


PROCESSED_ROOT = DATA_ROOT / "processed"
PHASE2_ROOT = PROCESSED_ROOT / "phase2"
TABLES_PHASE2_ROOT = PROJECT_ROOT / "tables" / "phase2"
LOGS_PHASE2_ROOT = TABLES_PHASE2_ROOT / "logs"
RESOURCES_PHASE2_ROOT = DATA_ROOT / "resources" / "phase2"

PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
LOGS_PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
