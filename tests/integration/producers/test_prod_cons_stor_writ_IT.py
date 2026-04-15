import json
import pytest
import logging
import asyncio

from stellar_harvest_ie_config.utils.log_decorators import log_io

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

from kafka import KafkaConsumer

import stellar_harvest_ie_stream.clients as stream_clients
from stellar_harvest_ie_stream.settings import settings as stream_settings
from stellar_harvest_ie_models.stellar.swpc.db import Base
from stellar_harvest_ie_models.stellar.swpc.models import KpIndexRecord
from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity

from stellar_harvest_ie_producers.stellar.swpc.producer import (
    publish_latest_planetary_kp_index,
)
from stellar_harvest_ie_consumers.stellar.swpc.service.kp_index_service import (
    KpIndexConsumerService,
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

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[kafka_bootstrap_server],
        group_id=f"test-{topic}-consumer",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    received: list[dict] = []
    try:
        for msg in consumer:
            received.append(msg.value)
            break  # one message is enough to validate the pipeline
    finally:
        consumer.close()

    assert received, f"No messages received on topic '{topic}'"

    single_raw_message = received[0]

    record = KpIndexRecord(**single_raw_message)

    assert record.time_tag is not None, "time_tag must be set"

    async def _store() -> KpIndexEntity:
        async with TestAsyncSessionLocal() as session:
            service = KpIndexConsumerService(session)
            return await service.create(single_raw_message)

    entity = asyncio.run(_store())

    assert entity.id is not None, "Entity must have a database-assigned id"

    async def _query() -> list[KpIndexEntity]:
        async with TestAsyncSessionLocal() as session:
            result = await session.execute(select(KpIndexEntity))
            return result.scalars().all()

    rows = asyncio.run(_query())

    assert len(rows) == 1, f"Expected exactly 1 row in kp_index table, got {len(rows)}"
