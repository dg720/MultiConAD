from pathlib import Path

from processing.phase1.common import DATA_ROOT, PROJECT_ROOT


PROCESSED_ROOT = DATA_ROOT / "processed"
PHASE2_ROOT = PROCESSED_ROOT / "phase2"
TABLES_PHASE2_ROOT = PROJECT_ROOT / "tables" / "02-ablation-task-specific-features"
TABLES_PHASE2_TABLES_ROOT = TABLES_PHASE2_ROOT / "result-tables"
TABLES_PHASE2_RESULT_TABLES = TABLES_PHASE2_TABLES_ROOT / "csv"
TABLES_PHASE2_SUMMARIES = TABLES_PHASE2_ROOT / "summaries"
LOGS_PHASE2_ROOT = TABLES_PHASE2_ROOT / "logs"
RESOURCES_PHASE2_ROOT = DATA_ROOT / "resources" / "phase2"

PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_PHASE2_TABLES_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_PHASE2_RESULT_TABLES.mkdir(parents=True, exist_ok=True)
TABLES_PHASE2_SUMMARIES.mkdir(parents=True, exist_ok=True)
LOGS_PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
