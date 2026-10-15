from collections.abc import Iterator
from typing import Final

import pytest

from litellm.responses import dispatch
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.responses.support import FallbackProbe
from tests.test_litellm_rust.support.isolation import rebound


@pytest.fixture
def native_responses() -> Iterator[FallbackProbe]:
    probe: Final = FallbackProbe()
    python_aresponses: Final = dispatch._PYTHON_ARESPONSES  # pyright: ignore[reportPrivateUsage]  # the fallback under observation
    python_responses: Final = dispatch._PYTHON_RESPONSES  # pyright: ignore[reportPrivateUsage]  # the fallback under observation

    async def python_fallback(*args: object, **kwargs: object) -> object:
        probe.calls += 1
        return await python_aresponses(*args, **kwargs)

    def python_sync_fallback(*args: object, **kwargs: object) -> object:
        probe.calls += 1
        return python_responses(*args, **kwargs)

    with (
        rebound(catalog, "RULES", (RouteRule(Route.RESPONSES, Rollout.RUST_OPT_IN), *catalog.RULES)),
        rebound(dispatch, "_PYTHON_ARESPONSES", python_fallback),
        rebound(dispatch, "_PYTHON_RESPONSES", python_sync_fallback),
    ):
        yield probe
