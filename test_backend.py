"""Integration tests for the multi-user API key auth backend."""
import sys
import os

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def run_tests():
    print("=" * 50)
    print("Multi-User Auth + Neo4j Integration Tests")
    print("=" * 50)

    # Test 1: Generate API Key via PostgreSQL
    print("\n[Test 1] Generate API Key from PostgreSQL")
    res = client.post("/v1/admin/generate_key", json={"owner_email": "test_run@babcock.edu"})
    if res.status_code != 200:
        print(f"  FAIL: {res.status_code} - {res.text}")
        return
    api_key = res.json()["api_key"]
    print(f"  PASS: Generated key = {api_key[:15]}...")

    # Test 2: Reject invalid key
    print("\n[Test 2] Reject Invalid API Key")
    r2 = client.post(
        "/v1/graph",
        json={"query": "MATCH (n) RETURN n LIMIT 1"},
        headers={"Authorization": "Bearer bu_kg_invalid_fake_key"},
    )
    print(f"  Status: {r2.status_code} (Expected: 401)")
    assert r2.status_code == 401, f"Expected 401, got {r2.status_code}"
    print("  PASS")

    # Test 3: Valid key -> Neo4j Graph Query
    print("\n[Test 3] Valid Key + Neo4j Graph Query")
    headers = {"Authorization": f"Bearer {api_key}"}
    cypher = "MATCH (s:School) RETURN s.Name AS SchoolName LIMIT 2"
    print(f"  Cypher: {cypher}")
    r3 = client.post("/v1/graph", json={"query": cypher}, headers=headers)
    assert r3.status_code == 200, f"Expected 200, got {r3.status_code}: {r3.text}"
    nodes = r3.json()["data"]["graph"]["nodes"]
    print(f"  PASS: {len(nodes)} nodes returned")
    for n in nodes:
        print(f"    -> {n['properties']}")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
