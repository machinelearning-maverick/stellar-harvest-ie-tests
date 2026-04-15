import pytest
from kafka.admin import KafkaAdminClient, NewTopic
from testcontainers.kafka import KafkaContainer
from testcontainers.postgres import PostgresContainer

from stellar_harvest_ie_config.logging_config import setup_logging

setup_logging()

from stellar_harvest_ie_stream.settings import settings as stream_settings


@pytest.fixture(scope="session")
def kafka_container():
    """
    Starts a Confluent cp-kafka 7.5.0 container (matching docker-compose.yml).
    ZooKeeper is managed internally by the KafkaContainer helper.
    """
    with KafkaContainer(image="confluentinc/cp-kafka:7.5.0") as container:
        yield container


@pytest.fixture(scope="session")
def kafka_bootstrap_server(kafka_container) -> str:
    """
    Extracts the bootstrap server address string, e.g. "localhost:PORT".
    Creates the swpc_topic explicitly since docker-compose has
    KAFKA_AUTO_CREATE_TOPICS_ENABLE=false.
    """
    bootstrap = kafka_container.get_bootstrap_server()

    admin = KafkaAdminClient(bootstrap_servers=[bootstrap])
    try:
        admin.create_topics(
            [
                NewTopic(
                    name=stream_settings.swpc_topic,
                    num_partitions=1,
                    replication_factor=1,
                )
            ]
        )
    finally:
        admin.close()

    return bootstrap


@pytest.fixture(scope="session")
def postgres_container():
    """
    Starts a postgres:16-alpine container (matching docker-compose.yml).
    """
    with PostgresContainer(image="postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_url(postgres_container) -> str:
    """
    Constructs an asyncpg-compatible connection URL from the PostgresContainer.
    """
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)

    return (
        f"postgresql+asyncpg://{postgres_container.username}:{postgres_container.password}"
        f"@{host}:{port}/{postgres_container.dbname}"
    )
