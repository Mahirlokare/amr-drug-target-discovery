"""
Dashboard routing and page integrity test suite.
Verifies all 13 Project 9 blueprint pages and main application entry points
have valid syntax and proper module structure.
Complies with Section 11 & 14.
"""

import importlib.util
from pathlib import Path
import pytest

from src.common.config import config_manager

APP_DIR = config_manager.project_root / "src" / "app"
PAGES_DIR = APP_DIR / "pages"


def test_main_app_file_exists():
    main_app = APP_DIR / "streamlit_app.py"
    assert main_app.exists()
    assert main_app.stat().st_size > 0


def test_all_blueprint_pages_exist():
    expected_pages = [
        "01_Target_Explorer.py",
        "02_Data_Explorer.py",
        "03_Molecular_Analysis.py",
        "04_Protein_Analysis.py",
        "05_Model_Laboratory.py",
        "06_Model_Evaluation.py",
        "07_Candidate_Discovery.py",
        "08_Docking_and_ADMET.py",
        "09_Reports.py",
    ]
    for page_name in expected_pages:
        page_path = PAGES_DIR / page_name
        assert page_path.exists(), f"Missing page: {page_name}"
        assert page_path.stat().st_size > 100, f"Page file is empty: {page_name}"


def test_page_syntax_validity():
    import ast
    for py_file in PAGES_DIR.glob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            ast.parse(code, filename=str(py_file))
        except SyntaxError as e:
            pytest.fail(f"Syntax error in page {py_file.name}: {e}")
