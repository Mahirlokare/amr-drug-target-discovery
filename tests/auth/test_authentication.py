"""
Authentication service and user management test suite.
Verifies PBKDF2 cryptographic hashing, constant-time verification,
user registration, authentication, and demo account seeding.
"""

import pytest
from src.auth.service import AuthService, hash_password, verify_password


@pytest.fixture
def auth_svc():
    return AuthService()


def test_password_hashing_and_verification():
    raw_pwd = "SecurePassword2026!"
    pwd_hash, salt = hash_password(raw_pwd)

    assert isinstance(pwd_hash, str) and len(pwd_hash) == 64  # SHA-256 hex
    assert isinstance(salt, str) and len(salt) == 32      # 16 bytes hex

    # Correct password verifies
    assert verify_password(raw_pwd, pwd_hash, salt) is True
    # Wrong password fails
    assert verify_password("WrongPassword!", pwd_hash, salt) is False


def test_seeded_accounts_authentication(auth_svc):
    # Admin
    ok_admin, msg_admin, u_admin = auth_svc.authenticate("admin", "adminpassword123")
    assert ok_admin is True
    assert u_admin is not None
    assert u_admin["username"] == "admin"
    assert "password_hash" not in u_admin
    assert "salt" not in u_admin

    # Scientist
    ok_sci, _, u_sci = auth_svc.authenticate("scientist", "amr2026pass")
    assert ok_sci is True
    assert u_sci["username"] == "scientist"

    # Reviewer
    ok_rev, _, u_rev = auth_svc.authenticate("reviewer", "reviewamr2026")
    assert ok_rev is True
    assert u_rev["username"] == "reviewer"


def test_authentication_failures(auth_svc):
    # Bad password
    ok_bad, msg_bad, _ = auth_svc.authenticate("admin", "incorrect_pwd")
    assert ok_bad is False
    assert "Incorrect password" in msg_bad

    # Unknown user
    ok_unknown, msg_unk, _ = auth_svc.authenticate("nonexistent_user", "somepassword")
    assert ok_unknown is False
    assert "Account not found" in msg_unk


def test_user_registration_and_validation(auth_svc):
    # Short password
    ok_short, msg_short, _ = auth_svc.register_user(
        username="testuser",
        email="test@univ.edu",
        password="123",
        full_name="Test User"
    )
    assert ok_short is False
    assert "at least 8 characters" in msg_short

    # Invalid email
    ok_email, msg_email, _ = auth_svc.register_user(
        username="testuser2",
        email="invalid-email-format",
        password="ValidPassword123!",
        full_name="Test User"
    )
    assert ok_email is False
    assert "valid institutional email" in msg_email

    # Successful registration
    import uuid
    uid_suffix = uuid.uuid4().hex[:6]
    new_username = f"watson_{uid_suffix}"
    new_email = f"watson_{uid_suffix}@bakerst.ac.uk"
    ok_reg, msg_reg, new_u = auth_svc.register_user(
        username=new_username,
        email=new_email,
        password="WatsonPassword2026!",
        full_name="Dr. John Watson",
        role="Clinical Microbiologist",
        institution="St. Bart's Hospital"
    )
    assert ok_reg is True
    assert new_u["username"] == new_username

    # Authenticate newly registered user
    ok_login, _, logged_u = auth_svc.authenticate(new_username, "WatsonPassword2026!")
    assert ok_login is True
    assert logged_u["full_name"] == "Dr. John Watson"

    # Duplicate username should fail
    ok_dup, msg_dup, _ = auth_svc.register_user(
        username=new_username,
        email=f"other_{uid_suffix}@bakerst.ac.uk",
        password="WatsonPassword2026!",
        full_name="Duplicate Watson"
    )
    assert ok_dup is False
    assert "already registered" in msg_dup
