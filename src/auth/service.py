"""
Authentication, cryptographic password hashing, and user account management.
Complies with Project 9 Blueprint Section 14.1 & Security Architecture.
"""

import hashlib
import hmac
import re
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.common.logging import get_logger
from src.storage.db import db_manager

logger = get_logger("auth")


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Computes secure PBKDF2-HMAC-SHA256 password hash with 100,000 iterations.
    Returns (password_hash, hex_salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)
    salt_bytes = bytes.fromhex(salt)
    pwd_bytes = password.encode("utf-8")
    derived = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, iterations=100000)
    return derived.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plain password against the stored PBKDF2 hash using constant-time comparison."""
    calculated_hash, _ = hash_password(password, salt=salt)
    return hmac.compare_digest(calculated_hash, stored_hash)


class AuthService:
    """Handles researcher registration, authentication, and session identity."""

    def __init__(self, db=None):
        self.db = db or db_manager
        self.seed_default_users()

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str,
        role: str = "researcher",
        institution: str = "AMR Research Consortium"
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validates credentials and registers a new researcher account.
        Returns: (success, message, user_dict)
        """
        clean_user = username.strip().lower()
        clean_email = email.strip().lower()

        # Validation rules
        if not re.match(r"^[a-zA-Z0-9_\-]{3,30}$", clean_user):
            return False, "Username must be 3-30 alphanumeric characters (underscores and hyphens allowed).", None

        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", clean_email):
            return False, "Please provide a valid institutional email address.", None

        if len(password) < 8:
            return False, "Password must be at least 8 characters long.", None

        # Check existing
        existing = self.db.get_user_by_username_or_email(clean_user)
        if existing:
            return False, f"Username '{clean_user}' is already registered.", None

        existing_email = self.db.get_user_by_username_or_email(clean_email)
        if existing_email:
            return False, f"Email '{clean_email}' is already associated with an account.", None

        pwd_hash, salt = hash_password(password)
        uid = f"USER-{uuid.uuid4().hex[:8].upper()}"

        user_data = {
            "user_id": uid,
            "username": clean_user,
            "email": clean_email,
            "password_hash": pwd_hash,
            "salt": salt,
            "full_name": full_name.strip() or clean_user,
            "role": role,
            "institution": institution.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None
        }

        self.db.save_user(user_data)
        logger.info(f"Registered new researcher account: {clean_user} ({uid})")

        safe_user = {k: v for k, v in user_data.items() if k not in ("password_hash", "salt")}
        return True, "Account registered successfully.", safe_user

    def authenticate(self, username_or_email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticates researcher credentials.
        Returns: (success, message, safe_user_dict)
        """
        clean_id = username_or_email.strip().lower()
        user = self.db.get_user_by_username_or_email(clean_id)

        if not user:
            return False, "Account not found. Please check your username or register.", None

        if not verify_password(password, user["password_hash"], user["salt"]):
            return False, "Incorrect password. Please verify your credentials.", None

        self.db.update_user_last_login(user["user_id"])
        logger.info(f"Successful sign-in: {user['username']} ({user['user_id']})")

        safe_user = {k: v for k, v in user.items() if k not in ("password_hash", "salt")}
        return True, f"Welcome back, {safe_user.get('full_name', safe_user['username'])}!", safe_user

    def seed_default_users(self) -> None:
        """Seeds benchmark researcher and auditor accounts if the users table is empty."""
        try:
            users = self.db.list_users()
            if users:
                return  # Accounts already exist

            defaults = [
                {
                    "username": "admin",
                    "email": "lead.investigator@amr-discovery.org",
                    "password": "adminpassword123",
                    "full_name": "Dr. Sarah Chen (Lead PI)",
                    "role": "Lead Investigator",
                    "institution": "Global AMR Discovery Consortium"
                },
                {
                    "username": "scientist",
                    "email": "bioinformatics@amr-discovery.org",
                    "password": "amr2026pass",
                    "full_name": "Dr. Alex Rivera (Bioinformatician)",
                    "role": "Senior Computational Chemist",
                    "institution": "Tuberculosis Drug Innovation Lab"
                },
                {
                    "username": "reviewer",
                    "email": "auditor@peer-review.org",
                    "password": "reviewamr2026",
                    "full_name": "Scientific Peer Reviewer",
                    "role": "Independent Validator",
                    "institution": "International Scientific Review Board"
                }
            ]

            for d in defaults:
                pwd_hash, salt = hash_password(d["password"])
                record = {
                    "user_id": f"USER-SEED-{d['username'].upper()}",
                    "username": d["username"],
                    "email": d["email"],
                    "password_hash": pwd_hash,
                    "salt": salt,
                    "full_name": d["full_name"],
                    "role": d["role"],
                    "institution": d["institution"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_login": None
                }
                self.db.save_user(record)

            logger.info("Seeded 3 default researcher accounts into database.")
        except Exception as e:
            logger.warning(f"Could not seed default accounts: {e}")


# Global Singleton instance
auth_service = AuthService()
