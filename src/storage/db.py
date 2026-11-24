"""
Relational Database Manager for AMR Drug Target Discovery.
Provides resilient storage with SQLite and PostgreSQL support, schema migrations,
CRUD operations, and health checks.
Complies with Blueprint Section 14.1, 14.2 & Section 21.D.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.config import config_manager
from src.common.logging import get_logger

logger = get_logger("database")


class DatabaseManager:
    """Universal SQLite / PostgreSQL database connection and CRUD manager."""

    def __init__(self, db_url: Optional[str] = None):
        self.raw_url = db_url or os.environ.get("DATABASE_URL")
        self.is_postgres = False
        self.db_path: Optional[Path] = None

        if self.raw_url and (
            self.raw_url.startswith("postgres://")
            or self.raw_url.startswith("postgresql://")
        ):
            # Render PostgreSQL connection string fix
            self.postgres_url = self.raw_url.replace("postgres://", "postgresql://", 1)
            self.is_postgres = True
            logger.info("Configured for PostgreSQL persistence.")
        else:
            # Persistent SQLite database path
            custom_sqlite_path = os.environ.get("SQLITE_DB_PATH")
            if custom_sqlite_path:
                self.db_path = Path(custom_sqlite_path)
            elif Path("/var/data").exists() and Path("/var/data").is_dir():
                # Render persistent disk mount point
                self.db_path = Path("/var/data/amr_discovery.db")
            else:
                self.db_path = config_manager.project_root / "data" / "amr_discovery.db"

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Configured for SQLite persistence at {self.db_path}")

        self.init_schema()

    def get_connection(self):
        """Returns a database connection with dictionary-like cursor capability."""
        if self.is_postgres:
            try:
                import psycopg2
                import psycopg2.extras
                return psycopg2.connect(self.postgres_url)
            except Exception as e:
                logger.warning(
                    f"PostgreSQL connection failed ({e}). Falling back to local SQLite."
                )
                self.is_postgres = False
                if not self.db_path:
                    self.db_path = config_manager.project_root / "data" / "amr_discovery.db"
                    self.db_path.parent.mkdir(parents=True, exist_ok=True)
                return self._get_sqlite_conn()
        else:
            return self._get_sqlite_conn()

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        assert self.db_path is not None
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, cursor, sql: str, params: Tuple = ()) -> None:
        """Executes query mapping parameter placeholders to PostgreSQL format if active."""
        if self.is_postgres:
            sql = sql.replace("?", "%s")
        cursor.execute(sql, params)

    def _upsert(self, cursor, table: str, pkey: str, data: Dict[str, Any]) -> None:
        """Dialect-agnostic upsert for SQLite and PostgreSQL."""
        cols = list(data.keys())
        vals = tuple(data.values())

        if self.is_postgres:
            cols_str = ", ".join(cols)
            placeholders = ", ".join(["%s"] * len(cols))
            update_clauses = [f"{c} = EXCLUDED.{c}" for c in cols if c != pkey]
            update_str = ", ".join(update_clauses) if update_clauses else f"{pkey} = EXCLUDED.{pkey}"
            sql = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders}) ON CONFLICT ({pkey}) DO UPDATE SET {update_str};"
            cursor.execute(sql, vals)
        else:
            cols_str = ", ".join(cols)
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT OR REPLACE INTO {table} ({cols_str}) VALUES ({placeholders});"
            cursor.execute(sql, vals)

    def init_schema(self) -> None:
        """Initializes tables for Target, Compound, Activity, ModelRun, Prediction, Docking, ADMET with indices."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS targets (
                    target_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    accession TEXT NOT NULL,
                    organism TEXT NOT NULL,
                    gene TEXT NOT NULL,
                    sequence_hash TEXT NOT NULL,
                    chembl_target_id TEXT,
                    pdb_id TEXT,
                    selection_rationale TEXT,
                    created_at TEXT NOT NULL
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS compounds (
                    compound_id TEXT PRIMARY KEY,
                    smiles_raw TEXT NOT NULL,
                    smiles_standard TEXT NOT NULL,
                    inchikey TEXT NOT NULL,
                    standardisation_status TEXT NOT NULL,
                    molecular_weight REAL,
                    logp REAL,
                    created_at TEXT NOT NULL
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_records (
                    record_id TEXT PRIMARY KEY,
                    compound_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    value_raw REAL NOT NULL,
                    units_raw TEXT NOT NULL,
                    pic50 REAL NOT NULL,
                    label INTEGER,
                    label_status TEXT NOT NULL,
                    year INTEGER,
                    provenance_uri TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(compound_id) REFERENCES compounds(compound_id),
                    FOREIGN KEY(target_id) REFERENCES targets(target_id)
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id TEXT PRIMARY KEY,
                    target_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    auroc REAL,
                    pr_auc REAL,
                    brier_score REAL,
                    ef_1pct REAL,
                    split_type TEXT,
                    created_at TEXT NOT NULL
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    compound_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    smiles TEXT NOT NULL,
                    calibrated_prob REAL NOT NULL,
                    uncertainty REAL NOT NULL,
                    is_uncertain INTEGER NOT NULL,
                    is_out_of_domain INTEGER NOT NULL,
                    decision_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS docking_runs (
                    docking_id TEXT PRIMARY KEY,
                    prediction_id TEXT,
                    compound_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    receptor_pdb TEXT NOT NULL,
                    binding_affinity REAL NOT NULL,
                    rmsd_to_control REAL,
                    pose_path TEXT,
                    created_at TEXT NOT NULL
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS admet_runs (
                    admet_id TEXT PRIMARY KEY,
                    compound_id TEXT NOT NULL,
                    mol_weight REAL,
                    logp REAL,
                    hbd INTEGER,
                    hba INTEGER,
                    tpsa REAL,
                    rotatable_bonds INTEGER,
                    lipinski_pass INTEGER,
                    veber_pass INTEGER,
                    qed REAL,
                    pains_alerts INTEGER,
                    decision_status TEXT,
                    created_at TEXT NOT NULL
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    institution TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                );
                """)

                # Create indices for query performance
                indices = [
                    "CREATE INDEX IF NOT EXISTS idx_compounds_inchikey ON compounds(inchikey);",
                    "CREATE INDEX IF NOT EXISTS idx_activity_compound ON activity_records(compound_id);",
                    "CREATE INDEX IF NOT EXISTS idx_activity_target ON activity_records(target_id);",
                    "CREATE INDEX IF NOT EXISTS idx_activity_endpoint ON activity_records(endpoint);",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_id);",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_prob ON predictions(calibrated_prob);",
                    "CREATE INDEX IF NOT EXISTS idx_docking_target ON docking_runs(target_id);",
                    "CREATE INDEX IF NOT EXISTS idx_admet_compound ON admet_runs(compound_id);",
                    "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
                    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);"
                ]
                for idx_stmt in indices:
                    cursor.execute(idx_stmt)

            logger.info("Database schema verification and initialization complete with indexed relations.")
        finally:
            conn.close()

    def health_check(self) -> Dict[str, Any]:
        """Runs a diagnostic healthcheck on database connectivity and table counts."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            _ = cursor.fetchone()

            # Inspect table counts
            counts = {}
            for table in [
                "targets",
                "compounds",
                "activity_records",
                "model_runs",
                "predictions",
                "docking_runs",
                "admet_runs",
                "users",
            ]:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                row = cursor.fetchone()
                counts[table] = row[0] if row else 0

            conn.close()
            return {
                "status": "HEALTHY",
                "engine": "PostgreSQL" if self.is_postgres else "SQLite",
                "location": str(self.db_path) if self.db_path else "Remote PostgreSQL",
                "table_counts": counts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                "status": "UNHEALTHY",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # --- Target CRUD ---
    def save_target(self, target_data: Dict[str, Any]) -> None:
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                record = {
                    "target_id": target_data["target_id"],
                    "name": target_data["name"],
                    "accession": target_data["accession"],
                    "organism": target_data["organism"],
                    "gene": target_data["gene"],
                    "sequence_hash": target_data["sequence_hash"],
                    "chembl_target_id": target_data.get("chembl_target_id"),
                    "pdb_id": target_data.get("pdb_id"),
                    "selection_rationale": target_data.get("selection_rationale"),
                    "created_at": now,
                }
                self._upsert(cursor, "targets", "target_id", record)
        finally:
            conn.close()

    def list_targets(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            self._query(cursor, "SELECT * FROM targets ORDER BY target_id ASC;")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # --- Prediction CRUD ---
    def save_prediction(self, pred: Dict[str, Any]) -> None:
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                record = {
                    "prediction_id": pred["prediction_id"],
                    "run_id": pred["run_id"],
                    "compound_id": pred["compound_id"],
                    "target_id": pred["target_id"],
                    "smiles": pred["smiles"],
                    "calibrated_prob": float(pred["calibrated_prob"]),
                    "uncertainty": float(pred["uncertainty"]),
                    "is_uncertain": int(pred.get("is_uncertain", False)),
                    "is_out_of_domain": int(pred.get("is_out_of_domain", False)),
                    "decision_status": pred.get("decision_status", "NEEDS_REVIEW"),
                    "created_at": now,
                }
                self._upsert(cursor, "predictions", "prediction_id", record)
        finally:
            conn.close()

    def list_predictions(
        self, target_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if target_id:
                self._query(
                    cursor,
                    "SELECT * FROM predictions WHERE target_id = ? ORDER BY calibrated_prob DESC LIMIT ?;",
                    (target_id, limit),
                )
            else:
                self._query(
                    cursor,
                    "SELECT * FROM predictions ORDER BY calibrated_prob DESC LIMIT ?;",
                    (limit,),
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # --- Docking CRUD ---
    def save_docking_run(self, docking_data: Dict[str, Any]) -> None:
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                record = {
                    "docking_id": docking_data["docking_id"],
                    "prediction_id": docking_data.get("prediction_id"),
                    "compound_id": docking_data["compound_id"],
                    "target_id": docking_data["target_id"],
                    "receptor_pdb": docking_data["receptor_pdb"],
                    "binding_affinity": float(docking_data["binding_affinity"]),
                    "rmsd_to_control": docking_data.get("rmsd_to_control"),
                    "pose_path": docking_data.get("pose_path"),
                    "created_at": now,
                }
                self._upsert(cursor, "docking_runs", "docking_id", record)
        finally:
            conn.close()

    def list_docking_runs(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            self._query(cursor, "SELECT * FROM docking_runs ORDER BY binding_affinity ASC;")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # --- ADMET CRUD ---
    def save_admet_run(self, admet_data: Dict[str, Any]) -> None:
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                record = {
                    "admet_id": admet_data["admet_id"],
                    "compound_id": admet_data["compound_id"],
                    "mol_weight": admet_data.get("mol_weight"),
                    "logp": admet_data.get("logp"),
                    "hbd": admet_data.get("hbd"),
                    "hba": admet_data.get("hba"),
                    "tpsa": admet_data.get("tpsa"),
                    "rotatable_bonds": admet_data.get("rotatable_bonds"),
                    "lipinski_pass": int(admet_data.get("lipinski_pass", 1)),
                    "veber_pass": int(admet_data.get("veber_pass", 1)),
                    "qed": admet_data.get("qed"),
                    "pains_alerts": int(admet_data.get("pains_alerts", 0)),
                    "decision_status": admet_data.get("decision_status", "PRIORITISED"),
                    "created_at": now,
                }
                self._upsert(cursor, "admet_runs", "admet_id", record)
        finally:
            conn.close()


    # --- User CRUD ---
    def save_user(self, user_data: Dict[str, Any]) -> str:
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                record = {
                    "user_id": user_data["user_id"],
                    "username": user_data["username"].strip().lower(),
                    "email": user_data["email"].strip().lower(),
                    "password_hash": user_data["password_hash"],
                    "salt": user_data["salt"],
                    "full_name": user_data.get("full_name", user_data["username"]),
                    "role": user_data.get("role", "researcher"),
                    "institution": user_data.get("institution", "Academic Research"),
                    "created_at": user_data.get("created_at", now),
                    "last_login": user_data.get("last_login")
                }
                self._upsert(cursor, "users", "user_id", record)
                return record["user_id"]
        finally:
            conn.close()

    def get_user_by_username_or_email(self, identifier: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            clean_id = identifier.strip().lower()
            sql = "SELECT * FROM users WHERE username = ? OR email = ? LIMIT 1;"
            self._query(cursor, sql, (clean_id, clean_id))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            sql = "SELECT * FROM users WHERE user_id = ? LIMIT 1;"
            self._query(cursor, sql, (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_user_last_login(self, user_id: str) -> None:
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                sql = "UPDATE users SET last_login = ? WHERE user_id = ?;"
                self._query(cursor, sql, (now, user_id))
        finally:
            conn.close()

    def list_users(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            self._query(cursor, "SELECT user_id, username, email, full_name, role, institution, created_at, last_login FROM users ORDER BY created_at ASC;")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# Global Singleton instance
db_manager = DatabaseManager()


def init_db() -> DatabaseManager:
    return db_manager


def check_db_health() -> Dict[str, Any]:
    return db_manager.health_check()
