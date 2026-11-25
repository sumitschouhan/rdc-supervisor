import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_DEMO", "1")
    from supervisor_template.api import app

    with TestClient(app) as c:
        yield c


def test_send_complete(client):
    r = client.post("/chat/send", json={"thread_id": "a1", "message": "Show me trades for AAPL"}).json()
    assert r["status"] == "complete"
    assert r["trace"][0]["agent"] == "mesh_agent"


def test_interrupt_resume_roundtrip(client):
    r = client.post("/chat/send", json={"thread_id": "a2", "message": "Show instruments in the sector"}).json()
    assert r["status"] == "needs_input" and r["response_schema"]["title"] == "ClarificationAnswer"
    r = client.post("/chat/resume", json={"thread_id": "a2", "answer": "Technology"}).json()
    assert r["status"] == "complete" and r["trace"][0]["rows"] == 2


def test_resume_without_interrupt_is_409(client):
    assert client.post("/chat/resume", json={"thread_id": "nope", "answer": "x"}).status_code == 409


def test_stream_sse(client):
    with client.stream("POST", "/chat/stream", json={"thread_id": "a3", "message": "Show my positions"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert nodes[:3] == ["start_turn", "supervisor", "metadata_agent"]
    assert events[-2]["node"] == "synthesizer" and events[-1]["type"] == "done"


def test_history(client):
    client.post("/chat/send", json={"thread_id": "a4", "message": "Show my positions"})
    msgs = client.get("/chat/history/a4").json()["messages"]
    assert msgs[0] == {"role": "human", "content": "Show my positions"}
