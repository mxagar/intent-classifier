"""Nox sessions for local verification."""

import nox


nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "intent_classifier", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytype", "intent_classifier")


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest")
