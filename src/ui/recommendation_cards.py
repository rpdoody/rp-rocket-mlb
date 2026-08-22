from __future__ import annotations

from page_utils import _american_to_implied_prob


def _short(full_name: str) -> str:
    return full_name.split()[-1] if full_name else ""


def _parse_american(raw) -> int | None:
    """Parse a signed American odds value without fabricating missing data."""
    try:
        value = str(raw).strip().replace("+", "")
        if value.lower() in {"", "—", "none", "nan"}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_american(value: int | None) -> str:
    if value is None:
        return "—"
    return f"+{value}" if value >= 0 else str(value)


def _build_game_recs(
    game: dict,
    espn_game: dict | None,
    projection,
    historical_data=None,
) -> dict:
    """Build ML, run-line, and total recommendations from one model distribution."""
    if not espn_game:
        return {}

    home_full = game.get("home_name", "")
    away_full = game.get("away_name", "")
    dist = getattr(projection, "distribution", None)
    home_prob = getattr(projection, "home_win_probability", 0.5)
    away_prob = getattr(projection, "away_win_probability", 0.5)
    recs: dict = {}

    ml_h = _parse_american(espn_game.get("ml_home"))
    ml_a = _parse_american(espn_game.get("ml_away"))
    if ml_h is not None and ml_a is not None:
        impl_h = _american_to_implied_prob(ml_h)
        impl_a = _american_to_implied_prob(ml_a)
        recs["ml"] = {
            "home": {
                "team": home_full,
                "odds_str": _format_american(ml_h),
                "impl": impl_h,
                "est_prob": home_prob,
                "edge": home_prob - impl_h,
            },
            "away": {
                "team": away_full,
                "odds_str": _format_american(ml_a),
                "impl": impl_a,
                "est_prob": away_prob,
                "edge": away_prob - impl_a,
            },
            "best": "home" if home_prob - impl_h >= away_prob - impl_a else "away",
        }

    spread_h = _parse_american(espn_game.get("spread_home"))
    spread_a = _parse_american(espn_game.get("spread_away"))
    if spread_h is not None and spread_a is not None:
        home_favorite = (
            ml_h < ml_a if ml_h is not None and ml_a is not None else spread_h > 0 and spread_a <= 0
        )

        if home_favorite:
            home_pick, away_pick = (
                f"{_short(home_full)} −1.5",
                f"{_short(away_full)} +1.5",
            )
        else:
            home_pick, away_pick = (
                f"{_short(home_full)} +1.5",
                f"{_short(away_full)} −1.5",
            )

        if dist is not None:
            if home_favorite:
                home_rl, away_rl, _push_rl = dist.run_line_probabilities(-1.5)
            else:
                away_rl, home_rl, _push_rl = dist.run_line_probabilities(-1.5)
        else:
            home_rl = 0.5
            away_rl = 0.5

        impl_h = _american_to_implied_prob(spread_h)
        impl_a = _american_to_implied_prob(spread_a)
        recs["rl"] = {
            "home": {
                "pick": home_pick,
                "odds_str": _format_american(spread_h),
                "impl": impl_h,
                "est_prob": home_rl,
                "edge": home_rl - impl_h,
            },
            "away": {
                "pick": away_pick,
                "odds_str": _format_american(spread_a),
                "impl": impl_a,
                "est_prob": away_rl,
                "edge": away_rl - impl_a,
            },
            "best": "home" if home_rl - impl_h >= away_rl - impl_a else "away",
        }

    try:
        posted = float(espn_game.get("over_under"))
    except (TypeError, ValueError):
        posted = None
    over_price = _parse_american(espn_game.get("over_odds"))
    under_price = _parse_american(espn_game.get("under_odds"))

    if (
        dist is not None
        and posted is not None
        and over_price is not None
        and under_price is not None
    ):
        raw_over, raw_under, _push_prob = dist.total_probabilities(posted)
        over_prob = max(0.20, min(0.80, raw_over))
        under_prob = max(0.20, min(0.80, raw_under))
        impl_over = _american_to_implied_prob(over_price)
        impl_under = _american_to_implied_prob(under_price)
        recs["ou"] = {
            "posted": posted,
            "exp_total": getattr(projection, "total_runs", 0),
            "over": {
                "pick": f"Over {posted}",
                "odds_str": _format_american(over_price),
                "impl": impl_over,
                "est_prob": over_prob,
                "edge": over_prob - impl_over,
            },
            "under": {
                "pick": f"Under {posted}",
                "odds_str": _format_american(under_price),
                "impl": impl_under,
                "est_prob": under_prob,
                "edge": under_prob - impl_under,
            },
            "best": "over" if over_prob - impl_over >= under_prob - impl_under else "under",
        }

    return recs


def _rec_card_html(label: str, side: dict, exp_info: str) -> str:
    """Render one market recommendation as an HTML block."""
    del label
    edge_pct = side["edge"] * 100
    if edge_pct > 3:
        color, badge = "#16a34a", "✅ BET"
    elif edge_pct > 0:
        color, badge = "#d97706", "➡ LEAN"
    else:
        color, badge = "#dc2626", "⛔ PASS"

    pick_text = _short(side["team"]) if side.get("team") else side.get("pick", "—")
    return (
        f'<div style="background:{color}18;border-left:4px solid {color};padding:8px 12px;'
        f'border-radius:0 6px 6px 0;margin-bottom:4px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<b style="font-size:0.88rem">{pick_text}</b>'
        f'<span style="background:{color};color:white;border-radius:6px;padding:1px 8px;'
        f'font-size:0.7rem;font-weight:700">{badge}</span></div>'
        f'<div style="font-size:0.78rem;color:#555;margin-top:2px">Odds: <b>{side["odds_str"]}</b>'
        f' &nbsp;|&nbsp; Edge: <b style="color:{color}">{edge_pct:+.1f}%</b></div>'
        f'<div style="font-size:0.73rem;color:#888">{exp_info}</div></div>'
    )


def _projection_summary(projection) -> str:
    """Compact, transparent context summary for a prediction card."""
    adjustments = projection.adjustments
    return (
        f"Projected score: {getattr(projection, 'away_runs', 0):.2f} away / {getattr(projection, 'home_runs', 0):.2f} home "
        f"· Park ×{adjustments.get('park_multiplier', 1.0):.2f} "
        f"· Weather ×{adjustments.get('weather_multiplier', 1.0):.2f}"
    )
