import pytest
import logging
import asyncio

from stellar_harvest_ie_config.utils.log_decorators import log_io

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

import stellar_harvest_ie_stream.clients as stream_clients
from stellar_harvest_ie_stream.settings import settings as stream_settings
from stellar_harvest_ie_models.stellar.swpc.db import Base
from stellar_harvest_ie_producers.stellar.swpc.producer import (
    publish_latest_planetary_kp_index,
)

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
def test_it(kafka_bootstrap_server, postgres_url, monkeypatch):
    topic = stream_settings.swpc_topic
    monkeypatch.setattr(stream_settings, "kafka_uri", kafka_bootstrap_server)
    monkeypatch.setattr(stream_clients, "_producer", None)

    # Like in the stellar_harvest_ie_store.db
    test_engine = create_async_engine(postgres_url, echo=True)
    TestAsyncSessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _setup_schema():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup_schema())

    publish_latest_planetary_kp_index()

    pass
