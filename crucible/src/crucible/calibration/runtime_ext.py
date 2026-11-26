"""Decision-model contracts of the E4 models (read by `crucible.calibration.runtime`). A contract is added only when
its checkpoint is registered, so the runtime never offers a model that does not exist."""
from __future__ import annotations

import numpy as np


def _f(d, k):
    return np.asarray(d[k], dtype=np.float64)


def _hour(h):
    h = np.floor(np.mod(h, 24.0))
    return np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)


PARTICIPATION_KINDS = ("issue_open", "issue_comment", "pr_open", "pr_review", "other")


def specs() -> dict:
    from .registry import list_models
    from .runtime import Input, Spec
    have = set(list_models())
    S = {}

    if "ad_exposure_response@1" in have:
        from .registry import read_card
        from .runtime import AdDecision
        card = read_card("ad_exposure_response")
        S["ad_response@2"] = Spec(
            "ad_response@2", "ad_exposure_response@1",
            use=("advertising: the probability that an individual visits an advertiser's offer during a campaign "
                 "period when the campaign is displayed to it (display or feed placements), as a fitted log-odds "
                 "shift of the individual's own organic visit probability, scaled by the probability that it "
                 "attends at least one of its displays; draw once per individual and campaign period"),
            output="ad_visit", model_class=AdDecision,
            inputs=(Input("organic_visit_probability", "the individual's probability of visiting the advertiser's "
                          "offer during the period without the campaign, from the world's own discovery mechanism "
                          "and the individual's state (needs, interests, history, contacts)", "probability",
                          "individual", 0, 1),
                    Input("attended_probability", "probability that the individual attends at least one of its "
                          "displays of the campaign in the period: runtime.at_least_one over per-display "
                          "probabilities from a registered attention decision model evaluated on each display's own "
                          "features (relevance in the shared space, repetition, familiarity, time); 0 when not "
                          "displayed", "probability", "individual", 0, 1)),
            build=lambda d, m, s: np.column_stack([np.log(np.clip(_f(d, "organic_visit_probability"), 1e-6, 1 - 1e-6))
                                                   - np.log1p(-np.clip(_f(d, "organic_visit_probability"), 1e-6,
                                                                       1 - 1e-6))]),
            params={"reference_attention": float(card.get("reference_attention", 1.0)),
                    "incremental_conversion_ratio": float(card.get("served_conversion_ratio", 1.0))},
            scope={"use": ["display_campaign", "feed_campaign"]},
            limits=("Criteo Uplift v2.1 (randomised eligibility as the instrument for display; advertisers, countries "
                    "and dates undisclosed; released 2018). Identified: the log-odds shift of a targeted display on "
                    "users Criteo's bidder reached, as a function of their organic propensity. NOT identified "
                    "(assumptions, own fidelity record advertising.attention_composition): the dependence on "
                    "attention, relevance and repetition (through the attention model and A_ref = 1, conservative: "
                    "an attended world display never exceeds a Criteo display's effect), frequency beyond 'at least "
                    "one attended display' (no wear-in or wear-out), carry-over between periods, ad content or "
                    "price. Never negative. Absolute levels come from the world's own organic propensities; the "
                    "shift is invariant to the publisher's sub-sampling only if that sampling depended on the outcome "
                    "or the arm, not on exposure. Not for search ads (use 'search_result' is refused). "
                    "params['incremental_conversion_ratio'] (2.76) = (ITT on conversions / ITT on visits) / "
                    "(conversions per visit in the control arm): it is the conversion of ad-caused visits relative to "
                    "organic visits ONLY under the untestable assumption that a display does not change the "
                    "conversion of individuals who would have visited anyway; otherwise it mixes the two effects and "
                    "overstates caused visits' conversion. A world whose purchase decision is not a probability (a "
                    "value comparison) may not apply it, and then states that its displayed visitors convert as "
                    "organic ones."),
            output_doc=("columns p_visit, p_visit_organic, p_incremental; sample() returns [visit, incremental] "
                        "from one coupled uniform"))

    if "session_length@1" in have:
        def b_sess(d, m, s):
            hs, hc = _hour(_f(d, "local_hour"))
            n = _f(d, "own_sessions_before")
            mean_y = np.where(n > 0, _f(d, "own_mean_log_session_minutes"), float(m.card["population_mean_y"]))
            cols = {"user_mean_log_minutes": mean_y, "user_log_sessions_before": np.log1p(n),
                    "user_log_rows_per_session": np.log1p(np.where(n > 0, _f(d, "own_mean_items_per_session"), 0.0)),
                    "active_days_7d": _f(d, "active_days_7d"),
                    "log_minutes_since_last_session": np.log1p(np.minimum(_f(d, "minutes_since_last_session"), 10080)),
                    "sessions_earlier_today": _f(d, "sessions_earlier_today"), "hour_sin": hs, "hour_cos": hc,
                    "weekend": _f(d, "weekend")}
            return np.column_stack([cols[c] for c in m.features])
        S["session_minutes@1"] = Spec(
            "session_minutes@1", "session_length@1",
            use=("attention allocation: minutes an individual spends watching in one feed session, given its earlier "
                 "sessions and the time of day (draw once per session; content-level choices stay per item)"),
            output="quantiles", build=b_sess,
            inputs=(Input("own_mean_log_session_minutes", "mean of log1p(minutes watched) over the individual's "
                          "sessions on earlier days (ignored when own_sessions_before = 0)", "log1p(minutes)",
                          "individual", 0),
                    Input("own_sessions_before", "the individual's sessions on earlier days", "count", "individual", 0),
                    Input("own_mean_items_per_session", "mean feed items reached per earlier session", "count",
                          "individual", 0),
                    Input("active_days_7d", "active days among the previous 7", "days", "individual", 0, 7, True),
                    Input("minutes_since_last_session", "minutes since the individual's previous session ended "
                          "(10080 when none)", "minutes", "individual", 0),
                    Input("sessions_earlier_today", "the individual's sessions earlier the same day", "count",
                          "individual", 0),
                    Input("local_hour", "local hour of the session start", "hour", "context", 0, 24),
                    Input("weekend", "1 on Saturdays and Sundays", "flag", "context", 0, 1)),
            limits=("KuaiRand-Pure sessions (30-minute gap) of 27,285 Kuaishou users, 2022-04-22..05-08; only plays of "
                    "the 7,583 pool videos are logged, so minutes are lower than real app time; no content inputs."),
            output_doc="quantiles of minutes watched in the session at the card's levels")

    if "creator_return@1" in have:
        def b_cr(d, m, s):
            cols = {"log_plays_author_before": np.log1p(_f(d, "own_creator_views_before")),
                    "log_plays_author_today": np.log1p(_f(d, "own_creator_views_today")),
                    "log_days_since_first_play": np.log1p(_f(d, "days_since_first_creator_view")),
                    "author_log_pop_7d": np.log1p(1000.0 * _f(d, "creator_views_7d") /
                                                  np.maximum(_f(d, "active_individuals_7d"), 1)),
                    "user_log_plays_7d": np.log1p(_f(d, "own_views_7d")),
                    "user_log_authors_7d": np.log1p(_f(d, "own_creators_7d")),
                    "follows_author": _f(d, "follows_creator"), "liked_author_today": _f(d, "liked_creator_today"),
                    "watch_fraction_today": np.minimum(_f(d, "mean_watch_fraction_today"), 3.0),
                    "weekend": _f(d, "weekend")}
            return np.column_stack([cols[c] for c in m.features])
        S["creator_return@1"] = Spec(
            "creator_return@1", "creator_return@1",
            use=("repeat consumption: probability that an individual who consumed a creator's item today consumes "
                 "another item of the same creator within 7 days (includes re-exposure by feeds)"),
            output="probability", build=b_cr,
            inputs=(Input("own_creator_views_before", "the individual's views of the creator's items on earlier days",
                          "count", "pair", 0),
                    Input("own_creator_views_today", "views of the creator's items today (>= 1)", "count", "pair", 1),
                    Input("days_since_first_creator_view", "days since the individual first viewed the creator",
                          "days", "pair", 0),
                    Input("creator_views_7d", "views of the creator's items by everyone in the previous 7 days",
                          "count", "item", 0),
                    Input("active_individuals_7d", "distinct individuals active on the platform in the previous 7 "
                          "days", "count", "context", 0),
                    Input("own_views_7d", "the individual's views in the previous 7 days", "count", "individual", 0),
                    Input("own_creators_7d", "distinct creators the individual viewed in the previous 7 days",
                          "count", "individual", 0),
                    Input("follows_creator", "1 if the individual follows the creator", "flag", "pair", 0, 1),
                    Input("liked_creator_today", "1 if the individual liked one of the creator's items today",
                          "flag", "pair", 0, 1),
                    Input("mean_watch_fraction_today", "mean watch time / duration over today's views of the "
                          "creator (capped at 3)", "fraction", "pair", 0),
                    Input("weekend", "1 on Saturdays and Sundays", "flag", "context", 0, 1)),
            limits=("KuaiRand-Pure, 2022-04-08..05-08: repeat consumption of a creator within 7 days is rare (0.2% of "
                    "units) because most creators have one pool video; discrimination is weak (held-out AUC about "
                    "0.5); the outcome mixes recommender re-exposure with choice."),
            output_doc="P(another view of the creator within 7 days)")

    if "customer_return@1" in have:
        def b_cu(d, m, s):
            n = _f(d, "orders_before")
            mon = _f(d, "month")
            cols = {"log_orders_before": np.log1p(n), "log_tenure_days": np.log1p(_f(d, "tenure_days")),
                    "log_recency_days": np.log1p(np.where(n > 0, _f(d, "recency_days"), 400.0)),
                    "log_mean_gap_days": np.log1p(np.where(n >= 2, _f(d, "mean_gap_days"), 400.0)),
                    "log_spend_today": np.log1p(_f(d, "spend_today")),
                    "log_spend_total": np.log1p(_f(d, "spend_total")),
                    "log_lines_today": np.log1p(_f(d, "lines_today")),
                    "log_products_today": np.log1p(_f(d, "products_today")),
                    "cancel_share": _f(d, "cancel_share"), "uk": _f(d, "home_market"),
                    "month_sin": np.sin(2 * np.pi * mon / 12), "month_cos": np.cos(2 * np.pi * mon / 12)}
            return np.column_stack([cols[c] for c in m.features])
        S["customer_return@1"] = Spec(
            "customer_return@1", "customer_return@1",
            use=("product retention: probability that a customer who ordered from a shop today orders again within "
                 "90 days, given its order history with that shop"),
            output="probability", build=b_cu,
            inputs=(Input("orders_before", "days with an order from this shop before today", "count", "pair", 0),
                    Input("tenure_days", "days since the customer's first order from the shop", "days", "pair", 0),
                    Input("recency_days", "days since the previous order (ignored when orders_before = 0)", "days",
                          "pair", 0),
                    Input("mean_gap_days", "mean days between earlier orders (ignored when orders_before < 2)",
                          "days", "pair", 0),
                    Input("spend_today", "money spent today with the shop, in GBP-equivalent units (the world's "
                          "currency divided by its declared GBP exchange factor)", "currency", "pair", 0),
                    Input("spend_total", "cumulative spend with the shop including today", "currency", "pair", 0),
                    Input("lines_today", "order lines today", "count", "pair", 0),
                    Input("products_today", "distinct products ordered today", "count", "pair", 0),
                    Input("cancel_share", "cancelled lines / all lines of the customer with the shop", "fraction",
                          "pair", 0, 1),
                    Input("home_market", "1 if the customer is in the shop's home market", "flag", "pair", 0, 1),
                    Input("month", "month of the year 1..12", "month", "context", 1, 12, True)),
            limits=("UCI Online Retail II: one UK online gift-ware retailer (many wholesalers), 2009-12..2011-12; "
                    "observational; no price, service or marketing inputs; 90-day horizon only; spend is in GBP of "
                    "2010-2011, so a world must declare its currency factor (no rescaling from the batch)."),
            output_doc="P(another order within 90 days)")

    if "product_demand@1" in have:
        def b_pd(d, m, s):
            w = _f(d, "week_of_year")
            cols = {"log_rel_price": np.log(_f(d, "price") / _f(d, "reference_price")),
                    "log_units_prev_week": np.log1p(_f(d, "units_prev_week")),
                    "log_units_prev_4w": np.log1p(_f(d, "mean_units_prev_4w")),
                    "log_customers_prev_4w": np.log1p(_f(d, "customers_prev_4w")),
                    "product_log_mean_units": _f(d, "product_mean_log_units"),
                    "week_sin": np.sin(2 * np.pi * w / 52.0), "week_cos": np.cos(2 * np.pi * w / 52.0),
                    "weeks_since_first_sale_log": np.log1p(_f(d, "weeks_since_first_sale"))}
            return np.column_stack([cols[c] for c in m.features])
        S["product_weekly_demand@1"] = Spec(
            "product_weekly_demand@1", "product_demand@1",
            use=("purchase and price response: expected log1p(units) a product sells in a week given its price "
                 "relative to its own usual price and its recent sales (an association, not a causal elasticity)"),
            output="log_units", build=b_pd,
            inputs=(Input("price", "the product's retail price this week", "currency", "item", 1e-9),
                    Input("reference_price", "the product's usual price (median over its earlier weeks)", "currency",
                          "item", 1e-9),
                    Input("units_prev_week", "units sold the previous week", "count", "item", 0),
                    Input("mean_units_prev_4w", "mean units over the previous 4 weeks", "count", "item", 0),
                    Input("customers_prev_4w", "distinct buyers in the previous 4 weeks", "count", "item", 0),
                    Input("product_mean_log_units", "mean of log1p(weekly units) over the product's earlier weeks",
                          "log1p(units)", "item", 0),
                    Input("week_of_year", "week of the year 0..51, week 0 = ISO week 49", "week", "context", 0, 52),
                    Input("weeks_since_first_sale", "weeks since the product's first sale", "weeks", "item", 0)),
            limits=("UCI Online Retail II, one UK online gift retailer, 2009-11..2011-12; retailer-set prices "
                    "(seasonal re-pricing, clearance) confound the price association; durable gift products only; no "
                    "competition or substitution."),
            output_doc="E[log1p(units)]; sample() adds a residual with the card's residual_sd and returns units")

    if "repo_participation_return@1" in have:
        def b_pr(d, m, s):
            k = _f(d, "first_kind").astype(int)
            hs, hc = _hour(_f(d, "utc_hour"))
            lg = lambda n: np.log1p(_f(d, n))      # noqa: E731
            cols = {"first_is_issue_open": (k == 0).astype(float), "first_is_issue_comment": (k == 1).astype(float),
                    "first_is_pr_open": (k == 2).astype(float), "first_is_pr_review": (k == 3).astype(float),
                    "repo_issue_events_24h": lg("community_issue_events_24h"),
                    "repo_pr_events_24h": lg("community_pr_events_24h"),
                    "repo_participants_24h": lg("community_participants_24h"),
                    "repo_owner_events_24h": lg("community_owner_events_24h"),
                    "repo_stars_24h": lg("community_endorsements_24h"),
                    "repo_pushes_24h": lg("community_pushes_24h"), "repo_org": _f(d, "community_org"),
                    "actor_events_24h": lg("own_events_24h"),
                    "actor_participation_repos_24h": lg("own_participation_communities_24h"),
                    "actor_repo_events_24h": lg("own_community_events_24h"), "hour_sin": hs, "hour_cos": hc}
            return np.column_stack([cols[c] for c in m.features])
        S["participation_return@1"] = Spec(
            "participation_return@1", "repo_participation_return@1",
            use=("community participation: probability that an individual who just took part in a project or "
                 "community discussion takes part there again within 24 h"),
            output="probability", build=b_pr,
            inputs=(Input("first_kind", f"kind of the participation, index into {list(PARTICIPATION_KINDS)}",
                          "category", "individual", 0, 4, True),
                    Input("community_issue_events_24h", "issue / question threads opened or commented in the "
                          "community in the previous 24 h", "count", "item", 0),
                    Input("community_pr_events_24h", "contribution proposals and reviews in the previous 24 h",
                          "count", "item", 0),
                    Input("community_participants_24h", "distinct participants in the previous 24 h", "count",
                          "item", 0),
                    Input("community_owner_events_24h", "events by the community's owner / maintainer in the previous "
                          "24 h", "count", "item", 0),
                    Input("community_endorsements_24h", "endorsements (stars, upvotes) in the previous 24 h", "count",
                          "item", 0),
                    Input("community_pushes_24h", "content updates (pushes) in the previous 24 h", "count", "item", 0),
                    Input("community_org", "1 if the community is run by an organisation", "flag", "item", 0, 1),
                    Input("own_events_24h", "the individual's events anywhere in the previous 24 h", "count",
                          "individual", 0),
                    Input("own_participation_communities_24h", "distinct communities the individual took part in "
                          "in the previous 24 h", "count", "individual", 0),
                    Input("own_community_events_24h", "the individual's events in this community in the previous "
                          "24 h", "count", "pair", 0),
                    Input("utc_hour", "hour of the participation (world clock)", "hour", "context", 0, 24)),
            limits=("GH Archive public GitHub issues and pull requests, 2024-10-07..10 UTC; software communities "
                    "only; 24 h horizon; observational; first participation is first within the 4-day window."),
            output_doc="P(another participation within 24 h)")

    if "star_cascade@1" in have:
        def b_sc(d, m, s):
            hs, hc = _hour(_f(d, "utc_hour"))
            lg = lambda n: np.log1p(_f(d, n))      # noqa: E731
            made = _f(d, "created_in_window") > 0
            cols = {"stars_prev_1h": lg("endorsements_prev_1h"), "stars_prev_3h": lg("endorsements_prev_3h"),
                    "stars_prev_24h": lg("endorsements_prev_24h"), "star_decay_2h": lg("endorsement_decay_2h"),
                    "star_decay_12h": lg("endorsement_decay_12h"), "forks_prev_24h": lg("forks_prev_24h"),
                    "events_prev_24h": lg("events_prev_24h"), "repo_org": _f(d, "owner_org"),
                    "created_in_window": made.astype(float),
                    "age_hours_log": np.log1p(np.where(made, _f(d, "age_hours"), 0.0)),
                    "global_stars_prev_1h": np.log1p(_f(d, "global_endorsements_prev_1h") / 1000.0),
                    "hour_sin": hs, "hour_cos": hc}
            return np.column_stack([cols[c] for c in m.features])
        S["endorsement_rate@1"] = Spec(
            "endorsement_rate@1", "star_cascade@1",
            use=("social diffusion: expected endorsements (stars, saves, upvotes) an item receives in the next hour "
                 "given its own recent endorsement history (self-exciting attention); for items with at least one "
                 "endorsement in the previous 24 h"),
            output="rate", build=b_sc,
            inputs=(Input("endorsements_prev_1h", "endorsements in the previous hour", "count", "item", 0),
                    Input("endorsements_prev_3h", "endorsements in the previous 3 hours", "count", "item", 0),
                    Input("endorsements_prev_24h", "endorsements in the previous 24 hours (>= 1 for the model's "
                          "units)", "count", "item", 0),
                    Input("endorsement_decay_2h", "sum over earlier hours of endorsements x exp(-age / 2 h)",
                          "count", "item", 0),
                    Input("endorsement_decay_12h", "the same with a 12 h decay", "count", "item", 0),
                    Input("forks_prev_24h", "copies / forks in the previous 24 h", "count", "item", 0),
                    Input("events_prev_24h", "all public events on the item in the previous 24 h", "count", "item", 0),
                    Input("owner_org", "1 if the owner is an organisation", "flag", "item", 0, 1),
                    Input("created_in_window", "1 if the item was created in the last 4 days", "flag", "item", 0, 1),
                    Input("age_hours", "hours since creation (used when created_in_window)", "hours", "item", 0),
                    Input("global_endorsements_prev_1h", "all endorsements on the platform in the previous hour",
                          "count", "context", 0),
                    Input("utc_hour", "hour of the world clock", "hour", "context", 0, 24)),
            limits=("GH Archive public GitHub stars, 2024-10-07..10 UTC; contagion, outside referrals and trending "
                    "feedback are not separated; one-day horizon; weekday only."),
            output_doc="expected endorsements next hour; sample() draws negative-binomial counts (card nb_alpha)")

    if "news_click_semantic@1" in have:
        def b_ns(d, m, s):
            sh = _f(d, "topic_population_share")
            cols = {"text_cos": _f(d, "text_cos"), "text_cos_z": _f(d, "text_cos_z"),
                    "cat_share": (_f(d, "own_topic_views") + sh) / (_f(d, "own_views_total") + 1.0),
                    "log_hist_len": np.log1p(_f(d, "own_views_total")),
                    "pop_log_clicks_per_1k": np.log1p(1000.0 * _f(d, "item_opens_24h") /
                                                      np.maximum(_f(d, "active_individuals_24h"), 1)),
                    "age_log_hours": np.log1p(_f(d, "item_age_hours")),
                    "slate_size_log": np.log1p(_f(d, "slate_size"))}
            return np.column_stack([cols[c] for c in m.features])
        S["article_open_semantic@1"] = Spec(
            "article_open_semantic@1", "news_click_semantic@1",
            use=("content selection: probability that an individual opens an article shown in a slate, with text "
                 "affinity measured in the shared semantic encoder space"),
            output="probability", build=b_ns,
            inputs=(Input("text_cos", "cosine between the item's text embedding (crucible.semantic.encoders.text, "
                          "MiniLM-L6, L2-normalised) and the L2-normalised mean embedding of the individual's consumed "
                          "items; 0 without history; a shared-space projection must preserve these cosines",
                          "cosine", "pair", -1, 1),
                    Input("text_cos_z", "text_cos standardised within the individual's OWN slate (mean and sd of "
                          "text_cos over the items of that slate; 0 when sd = 0) - the individual's decision context, "
                          "never other individuals' items", "z-score", "slate"),
                    Input("own_topic_views", "the individual's past views in the item's topic", "count", "pair", 0),
                    Input("own_views_total", "the individual's past views", "count", "individual", 0),
                    Input("topic_population_share", "share of all past views in the item's topic", "fraction",
                          "context", 0, 1),
                    Input("item_opens_24h", "opens of the item in the previous 24 h", "count", "item", 0),
                    Input("active_individuals_24h", "individuals active on the platform in the previous 24 h", "count",
                          "context", 0),
                    Input("item_age_hours", "hours since the item was first shown", "hours", "item", 0),
                    Input("slate_size", "items in the slate", "count", "slate", 1)),
            limits=("MIND-small (US Microsoft News, 2019-11-09..15), production-recommender exposure (observational); "
                    "news articles only; no dwell time; non-commercial licence; mirror copy used for "
                    "experimental research by owner decision (D-28)."),
            scope={"content_kinds": ["article", "newsletter"]},
            output_doc="P(open) per slot")
    return S
