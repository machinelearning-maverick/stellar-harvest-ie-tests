"""
Integration test: SWPC KP-Index producer → Kafka → consumer service → PostgreSQL.

What this test proves end-to-end (no mocks):
  1. A real HTTP GET is issued to the NOAA SWPC API.
  2. The parsed KpIndexRecord is published to a real Kafka broker
     (managed by pytest-kafka / ZooKeeper in-process).
  3. A real KafkaConsumer reads the message back from the topic.
  4. The raw Kafka payload is validated against the KpIndexRecord Pydantic model.
  5. KpIndexConsumerService persists the record via AsyncRepository into a real
     PostgreSQL database (managed by pytest-postgresql / pg_ctl in-process).
  6. A direct SELECT query confirms the row is stored with the expected values.

Fixtures used:
  kafka_server  – str, e.g. "localhost:PORT"    (from conftest.py / testcontainers KafkaContainer)
  postgresql    – PostgresContainer instance    (from conftest.py / testcontainers PostgresContainer)
  monkeypatch   – built-in pytest fixture
"""

import json
import asyncio
import pytest

from kafka import KafkaConsumer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

# -- Project imports --------------------------------------------------------
# Importing KpIndexEntity also registers it on Base (via the store → models
# import chain), which is required before create_all runs.
from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity
from stellar_harvest_ie_models.stellar.swpc.db import Base
from stellar_harvest_ie_models.stellar.swpc.models import KpIndexRecord

import stellar_harvest_ie_stream.clients as stream_clients
from stellar_harvest_ie_stream.settings import settings as stream_settings

from stellar_harvest_ie_producers.stellar.swpc.producer import (
    publish_latest_planetary_kp_index,
)
from stellar_harvest_ie_consumers.stellar.swpc.service.kp_index_service import (
    KpIndexConsumerService,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAFKA_CONSUMER_TIMEOUT_MS = 20_000  # 20 s – NOAA + network latency headroom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_asyncpg_url(postgresql) -> str:
    """Construct an asyncpg-compatible URL from a testcontainers PostgresContainer."""
    host = postgresql.get_container_host_ip()
    port = postgresql.get_exposed_port(5432)
    return (
        f"postgresql+asyncpg://{postgresql.username}:{postgresql.password}"
        f"@{host}:{port}/{postgresql.dbname}"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_publish_kp_index_produces_to_kafka_and_stores_to_postgres(
    kafka_server,
    postgresql,
    monkeypatch,
):
    """Full pipeline: NOAA HTTP → Kafka → consumer service → PostgreSQL."""

    topic = stream_settings.swpc_topic

    # -----------------------------------------------------------------------
    # 1. Redirect the producer to the test Kafka broker.
    #    monkeypatch.setattr is NOT a mock – it sets a real attribute value.
    #    We also reset the module-level singleton so a fresh KafkaProducer is
    #    created that connects to the test broker instead of the default one.
    # -----------------------------------------------------------------------
    monkeypatch.setattr(stream_settings, "kafka_uri", kafka_server)
    monkeypatch.setattr(stream_clients, "_producer", None)

    # -----------------------------------------------------------------------
    # 2. Build a test-scoped async SQLAlchemy engine pointing at the
    #    pytest-postgresql instance.  We deliberately do NOT touch the global
    #    engine in stellar_harvest_ie_store.db so the module remains unaffected
    #    after the test.
    # -----------------------------------------------------------------------
    db_url = _build_asyncpg_url(postgresql)
    test_engine = create_async_engine(db_url, echo=True)
    TestSessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create the kp_index table in the ephemeral PostgreSQL database.
    # KpIndexEntity is already registered on Base (imported above).
    async def _setup_schema() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup_schema())

    # -----------------------------------------------------------------------
    # 3. Call the producer.
    #    - Issues a real GET request to https://services.swpc.noaa.gov/…
    #    - Parses the response into a KpIndexRecord
    #    - Sends it to the test Kafka broker
    # -----------------------------------------------------------------------
    publish_latest_planetary_kp_index()

    # -----------------------------------------------------------------------
    # 4. Consume from the test Kafka topic with a real KafkaConsumer.
    #    consumer_timeout_ms causes the iterator to raise StopIteration (and
    #    exit the for-loop) if no new messages arrive within the timeout window.
    # -----------------------------------------------------------------------
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[kafka_server],
        group_id="integration-test-kp-index",
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=KAFKA_CONSUMER_TIMEOUT_MS,
        enable_auto_commit=True,
    )

    received: list[dict] = []
    try:
        for msg in consumer:
            received.append(msg.value)
            break  # one message is enough to validate the pipeline
    finally:
        consumer.close()

    assert received, (
        f"No messages received on topic '{topic}' within "
        f"{KAFKA_CONSUMER_TIMEOUT_MS / 1000:.0f} s. "
        "The producer may have failed or the HTTP request timed out."
    )

    raw_message = received[0]

    # -----------------------------------------------------------------------
    # 5. Validate the Kafka payload against the Pydantic model.
    # -----------------------------------------------------------------------
    record = KpIndexRecord(**raw_message)
    assert record.time_tag is not None, "time_tag must be set"
    assert isinstance(record.kp_index, int), "kp_index must be an int"
    assert isinstance(record.estimated_kp, float), "estimated_kp must be a float"
    assert isinstance(record.kp, str) and record.kp, "kp must be a non-empty string"

    # -----------------------------------------------------------------------
    # 6. Persist the message via the consumer service (real async SQLAlchemy).
    # -----------------------------------------------------------------------
    async def _store() -> KpIndexEntity:
        async with TestSessionLocal() as session:
            service = KpIndexConsumerService(session)
            return await service.create(raw_message)

    entity = asyncio.run(_store())

    assert entity.id is not None, "Entity must have a database-assigned id"
    assert entity.kp_index == record.kp_index
    assert entity.estimated_kp == pytest.approx(record.estimated_kp)
    assert entity.kp == record.kp

    # -----------------------------------------------------------------------
    # 7. Confirm the row is queryable from PostgreSQL.
    # -----------------------------------------------------------------------
    async def _query() -> list[KpIndexEntity]:
        async with TestSessionLocal() as session:
            result = await session.execute(select(KpIndexEntity))
            return result.scalars().all()

    rows = asyncio.run(_query())

    assert len(rows) == 1, f"Expected exactly 1 row in kp_index table, got {len(rows)}"
    assert rows[0].kp_index == record.kp_index
    assert rows[0].kp == record.kp
