import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from stellar_harvest_ie_models.base import Base
import stellar_harvest_ie_models.stellar.swpc.entities  # noqa: F401 — registers KpIndexEntity with Base


@pytest.fixture(scope="session")
def ml_db_session_factory(postgres_url):
    engine = create_async_engine(postgres_url, poolclass=NullPool)

    async def _create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def patch_loader(ml_db_session_factory, monkeypatch):
    monkeypatch.setattr(
        "stellar_harvest_ie_ml_stellar.data.loader.AsyncSessionLocal",
        ml_db_session_factory,
    )
