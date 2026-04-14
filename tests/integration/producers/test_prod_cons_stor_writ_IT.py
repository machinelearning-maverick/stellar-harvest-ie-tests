import pytest
import logging

from stellar_harvest_ie_config.utils.log_decorators import log_io

logger = logging.getLogger(__name__)


@log_io()
def _build_asyncpg_url(postgresql) -> str:
    """Construct an asyncpg-compatible URL from a testcontainers PostgresContainer."""
    host = postgresql.get_container_host_ip()
    port = postgresql.get_exposed_port(5432)
    return (
        f"postgresql+asyncpg://{postgresql.username}:{postgresql.password}"
        f"@{host}:{port}/{postgresql.dbname}"
    )


@log_io()
def test_it(kafka_server, postgresql, monkeypatch):
    db_url = _build_asyncpg_url(postgresql)

    pass
