"""Provider name -> connector class."""

from __future__ import annotations

from agentrs.harness.connectors.base import (
    ConnectorContext,
    ConnectorError,
    PlatformConnector,
)
from agentrs.harness.connectors.bluesky import BlueskyConnector
from agentrs.harness.connectors.discord import DiscordConnector
from agentrs.harness.connectors.generic_web import GenericWebConnector
from agentrs.harness.connectors.github import GitHubConnector
from agentrs.harness.connectors.instagram import InstagramConnector
from agentrs.harness.connectors.linkedin import LinkedInConnector
from agentrs.harness.connectors.mastodon import MastodonConnector
from agentrs.harness.connectors.reddit import RedditConnector
from agentrs.harness.connectors.socialweblab import SocialWebLabConnector
from agentrs.harness.connectors.x import XConnector

CONNECTORS: dict[str, type[PlatformConnector]] = {
    "bluesky": BlueskyConnector,
    "mastodon": MastodonConnector,
    "github": GitHubConnector,
    "discord": DiscordConnector,
    "x": XConnector,
    "instagram": InstagramConnector,
    "reddit": RedditConnector,
    "linkedin": LinkedInConnector,
    "socialweblab": SocialWebLabConnector,
    "web": GenericWebConnector,
}


def make_connector(provider: str, ctx: ConnectorContext) -> PlatformConnector:
    cls = CONNECTORS.get(provider)
    if cls is None:
        raise ConnectorError(f"no connector for provider {provider!r}", 404)
    if ctx.config.provider != provider:
        raise ConnectorError(f"context is for {ctx.config.provider!r}, not {provider!r}", 400)
    return cls(ctx)
