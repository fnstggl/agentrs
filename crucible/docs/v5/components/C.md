# Component C: the social internet on the one population (`crucible.social`, wave 3b)

Workstream C. Requirements: C.1-C.6, C.n1, D.1-D.5, I.1-I.7, I.n1, SP-4, SP-4.1b, SP-4.2, SP-4.5, SP-4.5b, SP-4.6,
SP-4.6b, SP-4.7, SP-4.7b, SP-4.8, SP-4.8b; inputs to E.3 (feed probes) and SP-5.5b (random exposure). Decisions D-33
(population media), D-34 (fitted open decision and random exposure on the short-video platform), D-35 (engagement
integrity, no observable representation). The audience accounting and the view threshold are in [C2.md](C2.md).

## What the population does on the platforms (`pop_social`)

| activity / hook | module | what members do (all through the platforms' own tools or internal APIs, as their own principal) |
|---|---|---|
| `social.create` | `social.popcreate` | choose a medium (post, status, article or newsletter issue, image card, narrated audio, narrated short or long slide video, Q&A answer/question on kb or dev, recommendation of a service that works for their job) by taste, ability and what earlier pieces of that medium earned; render it (`social.popmedia`, recorded) and publish it |
| `social.inbox` | `social.popinbox` | read messages, open linked items (they go to the top of the item's platform feed), answer, recommend the last liked item to the best-fitting contact; review paid memberships |
| `social.community` | `social.popsocial` | visit, read, post, reply, report, moderate, leave, found communities on subjects from what they read |
| `feed.<platform>` + after-session hook | `feed`, `social.popinbox` | browse feeds (consumption engine); then subscribe to newsletters, join paid memberships, vote and accept answers on Q&A platforms, unsubscribe, and look into a service for their own work that they read about (an `economy.work` `shop` session) |
| launch after-session | `social.popsocial` | visit launched products, buy, use, rate |

Columns: `pop_social` declares its per-medium columns only when a world offers more than one medium and its inbox
columns only when it has messaging or newsletters (the 10M-identity internet profile keeps three columns).

## The short-video platform's fitted open decision and random exposure (`social.contentopen`)

`crucible.social.worlds.feed_config(platforms)` (used by `social_world`; civilisations should use it too) sets for
`shorts`: `calibrated: {"open": "content_open@1", "content_kind": "video_short"}` and
`random_insert: {"rate": 0.05, "max_age_days": 30}`. The engine evaluates `content_open@1` through its scoped runtime
handle on every reached slot (inputs from day-start snapshots in the feed-owned column tables `feed_co_shorts_v` /
`feed_co_shorts_i`, process `feed.daystart`) and records the decisions per arm as `open.shorts` batches on the
population table (provenance `content_open@1`, columns `opened`, `own_topic`, `extra.experiment = {"id":
"random_exposure", "arm": "random" | "recommended"}`) inside the population's session, which is what
`crucible.calibration.inworld.random_exposure_experiment` reads.

## Item metadata has one owner

For catalogue items, `aud_item_info` holds only audience attributes (content id, creator code, measured duration,
search terms, word count, removed); use `crucible.social.audience.info(world, item)` for the merged view and
`audience.items_of(world, principal)` for a creator's items. `aud.check_invariants` refuses a copy of the catalogue's
metadata. Items without a catalogue record (the crowd stand-in's scale path) keep their metadata in the row.

## Follows have one owner: the social graph

With the one population and the social network installed (`audience.graph_follows(world)`: every social and
civilisation world), a follow is an edge of `social.cols:follow` and nowhere else (D-21 relationship migration,
audience side): the consumption engine writes the follows it decides through `graph.follow_batch`, the recommenders
read them through `recsys.features.follows_matrix`, and `aud` keeps no follower bitsets (`audset.fo.*`) and declares no
`aud_creators.followers` column; `audience.followers` / `audience.is_follower` are views of the graph. The audience
module still counts follow DECISIONS made after consuming an item (`subscribers` per item, `follows` per creator:
analytics, recounted per decision batch). Only the crowd stand-in's worlds (no `pop` or no `social`) keep the old
follower sets. Still outside this package: `forum.forum_follows` (the 4.0 forum's follows; forum should write social
edges where `social` is installed), `pop.cols:pop.contacts` (workstream B) and, for communities, `forum.forum_boards`
and `pop.cols:pop.communities` (see C.json integration notes).

## Security

`social.render.run_tool` (studio FFmpeg/SoX, and the version queries of the tool-chain identity) runs every native
tool through `crucible.sandbox.isolation.run`; `social.formats.image_info` and `social.render.open_image` decode
participant images with `isolation.decode_image` (only raw pixels enter the world process); Ogg audio is probed by
the jailed ffprobe. `social.integrity` excludes coordinated clusters (reciprocal-follow rings of three or more agents
engaging a target without independent attention to their own work). `aud.item` / `aud.item_updated` are internal
events: no participant observes an item's representation.

## E.3 probes (`social.probes`)

`social_feed_image` (image platform): 6,000 members browse the image platform's feed for three simulated days among
poster cards of six real news pieces plus the probed image; the outcome is seconds of consumption of the probed item
per individual. Both arms differ only in the artifact's bytes. The same world on the text-post platform
(`probes.make_world(..., platform="posts")`) is not registered: workstream D's `space_feed` (semantic.feedprobe)
measures the text-post feed. Without common random numbers between the arms (the engine's draws follow the event
positions), the per-individual noise is large; the reading probe (`population_reading`) remains the most powerful.
