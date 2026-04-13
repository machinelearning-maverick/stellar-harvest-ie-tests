import pytest
from kafka.admin import KafkaAdminClient, NewTopic
from testcontainers.kafka import KafkaContainer
from testcontainers.postgres import PostgresContainer

from stellar_harvest_ie_stream.settings import settings as stream_settings

# ---------------------------------------------------------------------------
# Same images used in docker-compose.yml — fresh containers, no shared state.
# Scoped to the session so they start once and are reused across all
# integration tests in the run.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def kafka_server():
    """
    Starts a Confluent cp-kafka 7.5.0 container (matching docker-compose.yml).
    ZooKeeper is managed internally by the KafkaContainer helper.
    Yields the bootstrap server address string, e.g. "localhost:PORT".
    """
    with KafkaContainer(image="confluentinc/cp-kafka:7.5.0") as kafka:
        bootstrap = kafka.get_bootstrap_server()

        # docker-compose has KAFKA_AUTO_CREATE_TOPICS_ENABLE=false, so we
        # must create the topic explicitly before any producer/consumer runs.
        admin = KafkaAdminClient(bootstrap_servers=[bootstrap])
        try:
            admin.create_topics([
                NewTopic(
                    name=stream_settings.swpc_topic,
                    num_partitions=1,
                    replication_factor=1,
                )
            ])
        finally:
            admin.close()

        yield bootstrap


@pytest.fixture(scope="session")
def postgresql():
    """
    Starts a postgres:16-alpine container (matching docker-compose.yml).
    Yields the PostgresContainer instance so tests can read connection details.
    """
    with PostgresContainer(image="postgres:16-alpine") as pg:
        yield pg
