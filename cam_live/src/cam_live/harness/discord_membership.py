"""cam_live.harness.discord_membership — the independent verifier's view of Discord membership. MASTER-OWNED.

``success.evidence_source: "cam_live.harness.discord_membership:membership_source"`` in a task file makes the
verifier process read the member list of the authorized guild itself, through the official bot REST API
(``GET /guilds/{guild.id}/members``, which needs the *Server Members* privileged intent), with **its own copy** of
the bot credential (``<run_dir>/verifier/secrets/<name>``, readable by the verifier uid only). Nothing here is
reachable by the learner: the verifier applies the task's exclusion rules to these records, hashes identifiers
before logging and releases aggregate counts only.

What the platform can and cannot tell the verifier:

* bot accounts (``user.bot`` / ``user.system``) — reported as bots;
* accounts that have not passed membership screening (``pending``) and guest members (member flag ``IS_GUEST``) —
  reported as pending;
* the account's creation time (from its snowflake id), so the task's ``min_account_age_s`` rule can apply;
* the server owner — the operator's own account — is not somebody who chose to join and is left out here;
* which invite a member used is **not** available from the REST API; ``invite_code`` stays empty.
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Callable, Dict, List, Optional

from ..contracts import TaskSpec
from .credentials import verifier_secret
from .discord import LIVE_BASE_URL, DiscordClient, environment_of

DISCORD_EPOCH_MS = 1420070400000
MEMBER_FLAG_IS_GUEST = 1 << 4


def snowflake_time(snowflake: str) -> float:
    """Creation time (unix seconds) encoded in a Discord snowflake id."""
    return ((int(snowflake) >> 22) + DISCORD_EPOCH_MS) / 1000.0


def _ts(iso: Optional[str]) -> float:
    if not iso:
        return 0.0
    return _dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def guild_settings(task: TaskSpec) -> Dict[str, Any]:
    r = dict(task.resources or {})
    d = dict(r.get("discord") or {})
    return {"guild_id": str(d.get("guild_id") or r.get("discord_guild_id") or ""),
            "token_secret": str(d.get("token_secret") or r.get("discord_bot_credential_name") or "discord_bot_token"),
            "base_url": str(d.get("base_url") or LIVE_BASE_URL)}


class DiscordMembershipSource:
    """``runtime.verifier.MembershipSource`` for one guild. ``live`` is true only against discord.com."""

    def __init__(self, guild_id: str, token_provider: Callable[[], Optional[str]], *, base_url: str = LIVE_BASE_URL,
                 client: Optional[DiscordClient] = None, page_size: int = 1000, max_pages: int = 1000):
        if not guild_id:
            raise ValueError("no Discord guild is configured for the verifier (resources.discord.guild_id)")
        self.guild_id, self.base_url = str(guild_id), base_url.rstrip("/")
        self.live = environment_of(self.base_url) == "live"
        self.name = "discord:bot-rest" if self.live else "discord:staging"
        self.page_size, self.max_pages = page_size, max_pages
        self._client = client or DiscordClient(self.base_url, token_provider,
                                               user_agent="DiscordBot (https://github.com/fnstggl/agentrs, 1.0)")
        self.last: Dict[str, Any] = {}

    def list_members(self) -> List[Any]:
        from ..runtime.verifier import MemberRecord
        c, g = self._client, self.guild_id
        owner = str(c._request("GET", f"/guilds/{g}", route=f"/guilds/{g}").get("owner_id") or "")
        out: List[Any] = []
        after, pages, owner_seen = "0", 0, False
        for _ in range(self.max_pages):
            page = c._request("GET", f"/guilds/{g}/members", route=f"/guilds/{g}/members",
                              params={"limit": str(self.page_size), "after": after})
            pages += 1
            for m in page:
                u = m.get("user") or {}
                uid = str(u["id"])
                if uid == owner:
                    owner_seen = True
                    continue
                guest = bool(int(m.get("flags") or 0) & MEMBER_FLAG_IS_GUEST)
                out.append(MemberRecord(id=uid, is_bot=bool(u.get("bot") or u.get("system")),
                                        joined_at=_ts(m.get("joined_at")),
                                        pending=bool(m.get("pending", False)) or guest,
                                        account_created_at=snowflake_time(uid)))
            if len(page) < self.page_size:
                break
            after = max((str(m["user"]["id"]) for m in page), key=int)
        self.last = {"listed": len(out), "pages": pages, "owner_excluded": owner_seen}
        return out

    def resource_usage(self) -> Dict[str, float]:
        return {"discord.verifier.requests": float(self._client.requests_sent)}


def membership_source(task: TaskSpec, run_dir: str) -> DiscordMembershipSource:
    """Factory named by ``success.evidence_source``; called inside the verifier process as ``factory(task, run_dir)``."""
    s = guild_settings(task)
    override = os.environ.get("CAM_LIVE_VERIFIER_SECRETS", "")

    def token() -> Optional[str]:
        if override:
            with open(os.path.join(override, s["token_secret"])) as f:
                return f.read().strip()
        return verifier_secret(run_dir, s["token_secret"])

    return DiscordMembershipSource(s["guild_id"], token, base_url=s["base_url"])
