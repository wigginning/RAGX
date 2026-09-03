"""Per-client-IP rate-limit ring (v1.0 security hardening, §9.2.4).

Regression: after middleware re-ordering, Auth rejects invalid credentials
BEFORE the per-key limiter runs, so a flood of bad keys was never throttled
(brute-force / DoS via 401 spam). ``IpRateLimitMiddleware`` sits outside
Auth and keys a token bucket on the peer IP; enabled only when
``SecurityConfig.ip_rate_limit_rps > 0`` (default off).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ragx.api.app import create_app
from ragx.core.settings import QueueConfig, SecurityConfig, Settings

GOOD_KEY = "sk-good-0001"


def _make_client(*, ip_rps: float = 0.0, ip_burst: int = 0, keys: bool = True) -> TestClient:
    api_keys = (
        {"a": {"key": GOOD_KEY, "kb_acl": [], "tenant_id": "t_a", "enabled": True}}
        if keys
        else {}
    )
    settings = Settings(
        queue=QueueConfig(auto_consume=False),
        security=SecurityConfig(
            api_keys=api_keys,
            rate_limit_rps=1000.0,  # per-key ring: effectively off in these tests
            rate_limit_burst=10000,
            ip_rate_limit_rps=ip_rps,
            ip_rate_limit_burst=ip_burst,
        ),
    )
    return TestClient(create_app(settings=settings))


def test_ip_ring_throttles_bad_credentials() -> None:
    """A flood of invalid keys is 401 at first, then 429 by IP."""
    bogus = {"Authorization": "Bearer sk-0000000000000000000000000000"}
    with _make_client(ip_rps=5.0, ip_burst=5) as c:
        codes = [c.get("/v1/health", headers=bogus).status_code for _ in range(6)]
    assert codes[:5] == [401] * 5, codes
    assert codes[5] == 429, "6th bad-credential request must be IP-throttled"


def test_ip_ring_throttles_open_endpoint_when_auth_off() -> None:
    """With auth disabled the IP ring still caps a single client."""
    with _make_client(ip_rps=5.0, ip_burst=5, keys=False) as c:
        codes = [c.get("/v1/health").status_code for _ in range(6)]
    assert codes[:5] == [200] * 5, codes
    assert codes[5] == 429


def test_ip_ring_disabled_by_default() -> None:
    """ip_rate_limit_rps defaults to 0 — no IP throttling unless opted in."""
    bogus = {"Authorization": "Bearer sk-0000000000000000000000000000"}
    with _make_client() as c:  # keys on, IP ring off
        codes = [c.get("/v1/health", headers=bogus).status_code for _ in range(8)]
    assert codes == [401] * 8, codes


def test_valid_and_invalid_share_the_ip_bucket() -> None:
    """IP tokens are spent by every request regardless of credential validity."""
    good = {"Authorization": f"Bearer {GOOD_KEY}"}
    bogus = {"Authorization": "Bearer sk-0000000000000000000000000000"}
    with _make_client(ip_rps=5.0, ip_burst=5) as c:
        seq = []
        for i in range(6):
            headers = good if i % 2 == 0 else bogus
            seq.append(c.get("/v1/health", headers=headers).status_code)
    # 5 tokens cover requests 0-4 (200 for the good key, 401 for bogus);
    # request 5 is throttled by IP before auth even looks at the key.
    assert seq == [200, 401, 200, 401, 200, 429], seq
