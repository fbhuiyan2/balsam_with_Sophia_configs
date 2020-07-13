from fastapi import status


def test_unauth_user_cannot_view_sites(anon_client):
    anon_client.get("/sites/", check=status.HTTP_401_UNAUTHORIZED)


def test_register(anon_client):
    login_credentials = {"username": "misha", "password": "foo"}
    resp = anon_client.post("/users/register", **login_credentials)
    assert type(resp["id"]) == int
    assert resp["username"] == login_credentials["username"]


def test_auth_user_can_view_sites(auth_client):
    resp = auth_client.get("/sites/")
    assert resp == []
