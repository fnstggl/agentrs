"""cam_live.harness.discord_perms — Discord permission bitfields, the operator's permission scope and pseudonyms.

Owned by workstream ES. Every number below is transcribed from the official API documentation
(github.com/discord/discord-api-docs, commit 62ab2d5, read 2026-11-27):

* ``PERMISSION_BITS`` — "Bitwise Permission Flags", ``developers/topics/permissions.mdx``.
* :func:`base_permissions` / :func:`channel_permissions` — the pseudocode of "Permission Overwrites" in the same
  document (``compute_base_permissions`` / ``compute_overwrites``).
* ``INTENT_BITS`` — "List of Intents", ``developers/events/gateway.mdx``.
* ``APPLICATION_INTENT_FLAGS`` — "Application Flags", ``developers/resources/application.mdx``.

Nothing here talks to anything; these are pure functions over what the adapter read.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: permission name -> bit position (``1 << n``)
PERMISSION_BITS: Dict[str, int] = {
    "CREATE_INSTANT_INVITE": 0, "KICK_MEMBERS": 1, "BAN_MEMBERS": 2, "ADMINISTRATOR": 3, "MANAGE_CHANNELS": 4,
    "MANAGE_GUILD": 5, "ADD_REACTIONS": 6, "VIEW_AUDIT_LOG": 7, "PRIORITY_SPEAKER": 8, "STREAM": 9,
    "VIEW_CHANNEL": 10, "SEND_MESSAGES": 11, "SEND_TTS_MESSAGES": 12, "MANAGE_MESSAGES": 13, "EMBED_LINKS": 14,
    "ATTACH_FILES": 15, "READ_MESSAGE_HISTORY": 16, "MENTION_EVERYONE": 17, "USE_EXTERNAL_EMOJIS": 18,
    "VIEW_GUILD_INSIGHTS": 19, "CONNECT": 20, "SPEAK": 21, "MUTE_MEMBERS": 22, "DEAFEN_MEMBERS": 23,
    "MOVE_MEMBERS": 24, "USE_VAD": 25, "CHANGE_NICKNAME": 26, "MANAGE_NICKNAMES": 27, "MANAGE_ROLES": 28,
    "MANAGE_WEBHOOKS": 29, "MANAGE_GUILD_EXPRESSIONS": 30, "USE_APPLICATION_COMMANDS": 31, "REQUEST_TO_SPEAK": 32,
    "MANAGE_EVENTS": 33, "MANAGE_THREADS": 34, "CREATE_PUBLIC_THREADS": 35, "CREATE_PRIVATE_THREADS": 36,
    "USE_EXTERNAL_STICKERS": 37, "SEND_MESSAGES_IN_THREADS": 38, "USE_EMBEDDED_ACTIVITIES": 39,
    "MODERATE_MEMBERS": 40, "VIEW_CREATOR_MONETIZATION_ANALYTICS": 41, "USE_SOUNDBOARD": 42,
    "CREATE_GUILD_EXPRESSIONS": 43, "CREATE_EVENTS": 44, "USE_EXTERNAL_SOUNDS": 45, "SEND_VOICE_MESSAGES": 46,
    "SET_VOICE_CHANNEL_STATUS": 48, "SEND_POLLS": 49, "USE_EXTERNAL_APPS": 50, "PIN_MESSAGES": 51,
    "BYPASS_SLOWMODE": 52,
}
ADMINISTRATOR = 1 << PERMISSION_BITS["ADMINISTRATOR"]
ALL_PERMISSIONS = 0
for _bit in PERMISSION_BITS.values():
    ALL_PERMISSIONS |= 1 << _bit

#: what the operator lets CAM use unless the task file says otherwise (permission integer 2269727275740401)
DEFAULT_SCOPE: Tuple[str, ...] = (
    "CREATE_INSTANT_INVITE", "MANAGE_CHANNELS", "MANAGE_GUILD", "ADD_REACTIONS", "VIEW_AUDIT_LOG", "VIEW_CHANNEL",
    "SEND_MESSAGES", "MANAGE_MESSAGES", "EMBED_LINKS", "ATTACH_FILES", "READ_MESSAGE_HISTORY", "MANAGE_ROLES",
    "MANAGE_EVENTS", "MANAGE_THREADS", "CREATE_PUBLIC_THREADS", "SEND_MESSAGES_IN_THREADS", "CREATE_EVENTS",
    "PIN_MESSAGES")
#: permissions the adapter never hands to a role, in a role's own bitfield or in a channel overwrite's ``allow``,
#: whatever the scope says: they would let the receiver act on members, on the server's settings or on everyone
NEVER_GRANTED: Tuple[str, ...] = ("ADMINISTRATOR", "KICK_MEMBERS", "BAN_MEMBERS", "MODERATE_MEMBERS", "MANAGE_GUILD",
                                  "MANAGE_ROLES", "MANAGE_WEBHOOKS", "MENTION_EVERYONE")

#: gateway intent name -> bit position
INTENT_BITS: Dict[str, int] = {
    "GUILDS": 0, "GUILD_MEMBERS": 1, "GUILD_MODERATION": 2, "GUILD_EXPRESSIONS": 3, "GUILD_INTEGRATIONS": 4,
    "GUILD_WEBHOOKS": 5, "GUILD_INVITES": 6, "GUILD_VOICE_STATES": 7, "GUILD_PRESENCES": 8, "GUILD_MESSAGES": 9,
    "GUILD_MESSAGE_REACTIONS": 10, "GUILD_MESSAGE_TYPING": 11, "DIRECT_MESSAGES": 12,
    "DIRECT_MESSAGE_REACTIONS": 13, "DIRECT_MESSAGE_TYPING": 14, "MESSAGE_CONTENT": 15,
    "GUILD_SCHEDULED_EVENTS": 16, "AUTO_MODERATION_CONFIGURATION": 20, "AUTO_MODERATION_EXECUTION": 21,
    "GUILD_MESSAGE_POLLS": 24, "DIRECT_MESSAGE_POLLS": 25,
}
PRIVILEGED_INTENTS: Tuple[str, ...] = ("GUILD_PRESENCES", "GUILD_MEMBERS", "MESSAGE_CONTENT")
#: privileged intent -> application flag bits that grant it (the full flag, the "limited" flag of small applications)
APPLICATION_INTENT_FLAGS: Dict[str, Tuple[int, int]] = {
    "GUILD_PRESENCES": (12, 13), "GUILD_MEMBERS": (14, 15), "MESSAGE_CONTENT": (18, 19)}


def bits_of(names: Iterable[str]) -> int:
    """Permission integer of a list of names. An unknown name is a ``ValueError`` (never silently dropped)."""
    out = 0
    for n in names:
        if n not in PERMISSION_BITS:
            raise ValueError(f"unknown Discord permission {n!r}")
        out |= 1 << PERMISSION_BITS[n]
    return out


def names_of(bits: Any) -> List[str]:
    """Sorted names of the documented permissions set in ``bits`` (an integer or its decimal string)."""
    v = int(bits or 0)
    return sorted(n for n, b in PERMISSION_BITS.items() if v >> b & 1)


def unknown_names(names: Iterable[str]) -> List[str]:
    return sorted({str(n) for n in names if n not in PERMISSION_BITS})


def normalise_scope(scope: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """The operator's scope as a tuple of valid names; ``None`` means the default scope."""
    if scope is None:
        return DEFAULT_SCOPE
    if isinstance(scope, str):
        raise ValueError("the Discord permission scope must be a list of permission names")
    bad = unknown_names(scope)
    if bad:
        raise ValueError(f"unknown Discord permission(s) in the operator scope: {', '.join(bad)}")
    return tuple(dict.fromkeys(str(n) for n in scope))


def base_permissions(guild_id: str, roles: Sequence[Mapping[str, Any]], member_role_ids: Iterable[str], *,
                     is_owner: bool = False) -> int:
    """Guild-level permissions of a member: the @everyone role (whose id is the guild id) OR-ed with the member's
    roles; ADMINISTRATOR (or owning the guild) means every permission."""
    if is_owner:
        return ALL_PERMISSIONS
    by_id = {str(r["id"]): int(r.get("permissions") or 0) for r in roles}
    perms = by_id.get(str(guild_id), 0)
    for rid in member_role_ids:
        perms |= by_id.get(str(rid), 0)
    if perms & ADMINISTRATOR == ADMINISTRATOR:
        return ALL_PERMISSIONS
    return perms


def channel_permissions(base: int, overwrites: Sequence[Mapping[str, Any]], guild_id: str,
                        member_role_ids: Iterable[str], member_id: str) -> int:
    """Apply a channel's permission overwrites to guild-level permissions, in the documented order: @everyone
    overwrite (deny, then allow), the member's role overwrites (all denies, then all allows), the member's own
    overwrite (deny, then allow). ADMINISTRATOR bypasses overwrites."""
    if base & ADMINISTRATOR == ADMINISTRATOR:
        return ALL_PERMISSIONS
    by_id = {str(o["id"]): o for o in overwrites or ()}
    perms = base
    everyone = by_id.get(str(guild_id))
    if everyone is not None:
        perms &= ~int(everyone.get("deny") or 0)
        perms |= int(everyone.get("allow") or 0)
    allow = deny = 0
    for rid in member_role_ids:
        o = by_id.get(str(rid))
        if o is not None and int(o.get("type", 0)) == 0:
            allow |= int(o.get("allow") or 0)
            deny |= int(o.get("deny") or 0)
    perms &= ~deny
    perms |= allow
    own = by_id.get(str(member_id))
    if own is not None and int(own.get("type", 1)) == 1:
        perms &= ~int(own.get("deny") or 0)
        perms |= int(own.get("allow") or 0)
    return perms


def timed_out_permissions(perms: int) -> int:
    """Permissions of a member while timed out: everything but ``VIEW_CHANNEL`` and ``READ_MESSAGE_HISTORY`` is
    lost; members with ``ADMINISTRATOR`` are exempt ("Permissions For Timed Out Members")."""
    if perms & ADMINISTRATOR == ADMINISTRATOR:
        return perms
    return perms & ((1 << PERMISSION_BITS["VIEW_CHANNEL"]) | (1 << PERMISSION_BITS["READ_MESSAGE_HISTORY"]))


def granted_intents(application_flags: Any) -> Dict[str, bool]:
    """Which privileged gateway intents the application's flags grant."""
    v = int(application_flags or 0)
    return {name: bool(v >> full & 1 or v >> limited & 1) for name, (full, limited) in APPLICATION_INTENT_FLAGS.items()}


def intents_value(names: Iterable[str]) -> int:
    out = 0
    for n in names:
        if n not in INTENT_BITS:
            raise ValueError(f"unknown gateway intent {n!r}")
        out |= 1 << INTENT_BITS[n]
    return out


def member_id_hash(guild_id: str, user_id: str, salt: str = "") -> str:
    """One-way pseudonym of a member, stable within a guild: enough to count distinct members and to see that the
    same member is still present later. It is never reversed."""
    return hashlib.sha256(f"{salt}:{guild_id}:{user_id}".encode()).hexdigest()[:32]
