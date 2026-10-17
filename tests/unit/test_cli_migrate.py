# Copyright 2026 Firefly Software Foundation
# SPDX-License-Identifier: Apache-2.0
"""Installed CLI migrations use the operator's reviewed working directory."""

from __future__ import annotations

import argparse

import pytest
from alembic import command

from flycanon import cli
from flycanon.config import get_settings


@pytest.fixture(autouse=True)
def _migration_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FLYCANON_DATABASE_URL", "postgresql+asyncpg://migration.invalid/flycanon")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_installed_migrate_uses_cwd_scaffold_and_configured_database(tmp_path, monkeypatch):
    runtime = tmp_path / "operator-runtime"
    runtime.mkdir()
    (runtime / "alembic.ini").write_text("[alembic]\nscript_location = migrations\n")
    monkeypatch.chdir(runtime)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "wheel/site-packages/flycanon/cli.py"))
    upgrades = []

    def upgrade(config, revision):
        upgrades.append(
            (
                config.config_file_name,
                config.get_main_option("script_location"),
                config.get_main_option("sqlalchemy.url"),
                revision,
            )
        )

    monkeypatch.setattr(command, "upgrade", upgrade)

    assert cli.cmd_migrate(argparse.Namespace()) == 0
    assert upgrades == [
        (
            str(runtime / "alembic.ini"),
            "migrations",
            "postgresql+asyncpg://migration.invalid/flycanon",
            "head",
        )
    ]


@pytest.mark.parametrize("config_is_directory", [False, True])
def test_missing_cwd_config_refuses_before_upgrade_without_searching_parents(
    tmp_path, monkeypatch, capsys, config_is_directory
):
    (tmp_path / "alembic.ini").write_text("[alembic]\nscript_location = migrations\n")
    runtime = tmp_path / "operator-runtime"
    runtime.mkdir()
    if config_is_directory:
        (runtime / "alembic.ini").mkdir()
    monkeypatch.chdir(runtime)

    def forbidden_upgrade(*args, **kwargs):
        pytest.fail("Missing migration scaffold must be refused before database migration")

    monkeypatch.setattr(command, "upgrade", forbidden_upgrade)

    assert cli.cmd_migrate(argparse.Namespace()) == 1
    error = capsys.readouterr().err
    assert str(runtime / "alembic.ini") in error
    assert "source archive" in error
    assert "working directory" in error
