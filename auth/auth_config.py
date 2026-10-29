from __future__ import annotations

import bcrypt
from sqlalchemy import or_
from sqlalchemy.orm import Session

from db.models import User
from db.session import SessionLocal


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def get_session() -> Session:
    return SessionLocal()


def ensure_user_exists(username: str, email: str, password: str) -> User:
    session = get_session()
    try:
        existing = session.query(User).filter((User.username == username) | (User.email == email)).first()
        if existing:
            raise ValueError("A user with that username or email already exists.")
        user = User(username=username, email=email, password_hash=hash_password(password))
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    finally:
        session.close()


def authenticate_user(username: str, password: str) -> User | None:
    session = get_session()
    try:
        identifier = username.strip()
        user = (
            session.query(User)
            .filter(or_(User.username == identifier, User.email == identifier.lower()))
            .first()
        )
        if not user:
            return None
        if bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            return user
        return None
    finally:
        session.close()
