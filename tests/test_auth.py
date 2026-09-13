import pytest
from fastapi.testclient import TestClient
from backend.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _token_for(client, user_id):
    res = client.post("/auth/token", json={"user_id": user_id})
    assert res.status_code == 200
    return res.json()["access_token"]


def test_ask_without_token_rejected(client):
    res = client.post("/ask", json={"query": "What is the parental leave policy?"})
    assert res.status_code == 401


def test_ask_with_garbage_token_rejected(client):
    res = client.post(
        "/ask",
        json={"query": "What is the parental leave policy?"},
        headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert res.status_code == 401


def test_login_unknown_user_rejected(client):
    res = client.post("/auth/token", json={"user_id": "totally_fake_user"})
    assert res.status_code == 401


def test_login_and_ask_flow(client):
    token = _token_for(client, "user_hr_mgr_01")

    perms_res = client.get("/debug/resolved-permissions", headers={"Authorization": f"Bearer {token}"})
    assert perms_res.status_code == 200
    perms = perms_res.json()
    assert perms["role"] == "HR_Manager"
    assert perms["user_id"] == "user_hr_mgr_01"

    ask_res = client.post(
        "/ask",
        json={"query": "What is the executive severance package?"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert ask_res.status_code == 200
    data = ask_res.json()
    assert data["resolved_permissions"]["user_id"] == "user_hr_mgr_01"
    assert data["answer"] != "I don't have access to information that answers this."


def test_cannot_impersonate_via_request_body(client):
    """
    Regression test for the identity-spoofing fix: AskRequest no longer has a
    user_id field, so a caller authenticated as a low-privilege persona cannot
    smuggle in a different identity by naming one in the JSON body.
    """
    token = _token_for(client, "user_support_01")

    ask_res = client.post(
        "/ask",
        json={"query": "What is the executive severance package?", "user_id": "user_vp_ops_01"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert ask_res.status_code == 200
    data = ask_res.json()
    assert data["resolved_permissions"]["user_id"] == "user_support_01"
    assert data["answer"] == "I don't have access to information that answers this."


def test_audit_log_requires_privileged_role(client):
    support_token = _token_for(client, "user_support_01")
    res = client.get("/audit-log", headers={"Authorization": f"Bearer {support_token}"})
    assert res.status_code == 403

    vp_token = _token_for(client, "user_vp_ops_01")
    res = client.get("/audit-log", headers={"Authorization": f"Bearer {vp_token}"})
    assert res.status_code == 200
    assert res.json()["integrity_check"]["valid"] is True


def test_audit_log_rejects_spoofed_header(client):
    """
    Regression test: the old X-User-Id header trust is gone. Sending it
    alongside a low-privilege token must NOT elevate access.
    """
    support_token = _token_for(client, "user_support_01")
    res = client.get(
        "/audit-log",
        headers={"Authorization": f"Bearer {support_token}", "X-User-Id": "user_vp_ops_01"}
    )
    assert res.status_code == 403


def test_tampered_token_rejected(client):
    token = _token_for(client, "user_hr_mgr_01")
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    res = client.get("/debug/resolved-permissions", headers={"Authorization": f"Bearer {tampered}"})
    assert res.status_code == 401
