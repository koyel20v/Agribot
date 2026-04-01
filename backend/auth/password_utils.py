# auth/password_utils.py — bcrypt password hashing helpers

import bcrypt


def hash_password(plain_password: str) -> str:
    """Hash a plain-text password. Returns the hashed string."""
    salt   = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if plain_password matches the stored hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )