"""
Persistence and Database Storage module for AMR Drug Target Discovery.
Supports SQLite (local / persistent volume) and PostgreSQL (production).
Complies with Project 9 Blueprint Section 14 (Storage & Database) and Section 21.D.
"""

from src.storage.db import (
    DatabaseManager,
    db_manager,
    init_db,
    check_db_health,
)

__all__ = ["DatabaseManager", "db_manager", "init_db", "check_db_health"]
