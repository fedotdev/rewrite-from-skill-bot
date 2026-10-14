import time

from core.deduplication import InMemoryDeduplicator


def test_deduplicator_first_time_not_duplicate():
    dedup = InMemoryDeduplicator()
    result = dedup.check_and_mark("key1")
    assert result.is_duplicate is False


def test_deduplicator_second_time_is_duplicate():
    dedup = InMemoryDeduplicator()
    dedup.check_and_mark("key1")
    result = dedup.check_and_mark("key1")
    assert result.is_duplicate is True


def test_deduplicator_different_keys():
    dedup = InMemoryDeduplicator()
    dedup.check_and_mark("key1")
    result = dedup.check_and_mark("key2")
    assert result.is_duplicate is False


def test_deduplicator_expires_after_ttl():
    dedup = InMemoryDeduplicator()
    dedup.check_and_mark("key1", ttl_seconds=1)
    time.sleep(1.1)
    result = dedup.check_and_mark("key1")
    assert result.is_duplicate is False


def test_deduplicator_cleanup_expired():
    dedup = InMemoryDeduplicator()
    dedup.check_and_mark("key1", ttl_seconds=1)
    dedup.check_and_mark("key2", ttl_seconds=10)
    time.sleep(1.1)
    cleaned = dedup.cleanup_expired()
    assert cleaned == 1
    # key2 should still exist
    result = dedup.check_and_mark("key2")
    assert result.is_duplicate is True
