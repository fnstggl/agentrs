"""OAuth 2.0 authorization-code flow with PKCE (S256) for connecting operator-approved accounts.

1. ``start`` builds the provider's authorization URL (state + PKCE challenge) for the **operator**
   to open in their browser; the agent never sees or completes the consent step.
2. The provider redirects to the control API's ``/oauth/callback``; ``complete`` checks the state
   (single use, 15-minute lifetime), exchanges the code through the egress-controlled client and
   writes the tokens to the secret store; the account record only receives secret references.
3. ``refresh`` renews access tokens; ``revoke`` calls the provider's revocation endpoint when it has one.
"""

from __future__ import annotations

import base64
import hashlib
import secrets as pysecrets
import time
from typing import Optional
from urllib.parse import urlencode

from agentrs.harness.db import Database, OAuthState
from agentrs.harness.identity import PlatformAccountRegistry
from agentrs.harness.net import SafeHTTPClient
from agentrs.harness.secrets import SecretStore

STATE_TTL_S = 900


class OAuthError(PermissionError):
    pass


class OAuthManager:
    def __init__(self, db: Database, secrets: SecretStore, accounts: PlatformAccountRegistry, backend: str = "file"):
        self.db, self.secrets, self.accounts, self.backend = db, secrets, accounts, backend

    def start(self, provider: str, account_id: str, authorize_url: str, client_id: str, redirect_uri: str,
              scopes: list[str], extra: Optional[dict] = None) -> str:
        verifier = base64.urlsafe_b64encode(pysecrets.token_bytes(48)).rstrip(b"=").decode()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = pysecrets.token_urlsafe(24)
        vref = self.secrets.put(self.backend if self.backend != "env" else "memory", f"pkce.{state}", verifier)
        with self.db.session() as s:
            s.add(OAuthState(state=state, provider=provider, account_id=account_id, code_verifier_ref=vref,
                             redirect_uri=redirect_uri, scopes=scopes))
        q = {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri, "scope": " ".join(scopes),
             "state": state, "code_challenge": challenge, "code_challenge_method": "S256", **(extra or {})}
        return f"{authorize_url}?{urlencode(q)}"

    async def complete(self, state: str, code: str, token_url: str, client_id: str, http: SafeHTTPClient,
                       client_secret_ref: Optional[str] = None) -> dict:
        with self.db.session() as s:
            st = s.get(OAuthState, state)
            if st is None or st.used or time.time() - st.created_at > STATE_TTL_S:
                raise OAuthError("unknown, used or expired OAuth state")
            st.used = True
            account_id, redirect_uri, vref, scopes = st.account_id, st.redirect_uri, st.code_verifier_ref, st.scopes
        data = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri, "client_id": client_id,
                "code_verifier": self.secrets.get(vref)}
        if client_secret_ref:
            data["client_secret"] = self.secrets.get(client_secret_ref)
        r = await http.post(token_url, data=data)
        self.secrets.delete(vref)
        if r.status_code != 200:
            raise OAuthError(f"token exchange failed: HTTP {r.status_code}")
        tok = r.json()
        refs = dict(self.accounts.get(account_id).credential_refs or {})
        refs["access_token"] = self.secrets.put(self.backend, f"{account_id}.access_token", tok["access_token"])
        if tok.get("refresh_token"):
            refs["refresh_token"] = self.secrets.put(self.backend, f"{account_id}.refresh_token", tok["refresh_token"])
        granted = (tok.get("scope") or " ".join(scopes)).replace(",", " ").split()
        exp = time.time() + float(tok["expires_in"]) if tok.get("expires_in") else None
        self.accounts.update(account_id, credential_refs=refs, oauth_scopes=granted, last_auth_at=time.time(),
                             reauth_by=exp)
        return {"account_id": account_id, "scopes": granted, "expires_at": exp}  # no token values returned

    async def refresh(self, account_id: str, token_url: str, client_id: str, http: SafeHTTPClient,
                      client_secret_ref: Optional[str] = None) -> dict:
        acc = self.accounts.get(account_id)
        rt = acc.credential_refs.get("refresh_token")
        if not rt:
            raise OAuthError("no refresh token; the operator must re-authorise")
        data = {"grant_type": "refresh_token", "refresh_token": self.secrets.get(rt), "client_id": client_id}
        if client_secret_ref:
            data["client_secret"] = self.secrets.get(client_secret_ref)
        r = await http.post(token_url, data=data)
        if r.status_code != 200:
            raise OAuthError(f"refresh failed: HTTP {r.status_code}")
        tok = r.json()
        refs = dict(acc.credential_refs)
        refs["access_token"] = self.secrets.put(self.backend, f"{account_id}.access_token", tok["access_token"])
        if tok.get("refresh_token"):
            refs["refresh_token"] = self.secrets.put(self.backend, f"{account_id}.refresh_token", tok["refresh_token"])
        exp = time.time() + float(tok["expires_in"]) if tok.get("expires_in") else None
        self.accounts.update(account_id, credential_refs=refs, last_auth_at=time.time(), reauth_by=exp)
        return {"account_id": account_id, "expires_at": exp}
