"""
tests/test_engine.py
====================
Unit tests for intervention/engine.py — RecommendationEngine.
"""

import pytest

from intervention.engine import (
    Intervention,
    RecommendationEngine,
    RecommendationPayload,
)


@pytest.fixture
def engine() -> RecommendationEngine:
    return RecommendationEngine()


# ---------------------------------------------------------------------------
# Layer 1: Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_suicide_triggers_crisis(self, engine):
        payload = engine.recommend("I want to commit suicide")
        assert payload.is_crisis is True
        assert payload.crisis_message is not None
        assert "988" in payload.crisis_message

    def test_kill_myself_triggers_crisis(self, engine):
        payload = engine.recommend("I want to kill myself")
        assert payload.is_crisis is True

    def test_end_it_all_triggers_crisis(self, engine):
        payload = engine.recommend("I just want to end it all")
        assert payload.is_crisis is True

    def test_self_harm_triggers_crisis(self, engine):
        payload = engine.recommend("I've been self-harming lately")
        assert payload.is_crisis is True

    def test_want_to_die_triggers_crisis(self, engine):
        payload = engine.recommend("I want to die")
        assert payload.is_crisis is True

    def test_crisis_halts_further_processing(self, engine):
        """When crisis is detected, only emergency interventions should be returned."""
        payload = engine.recommend(
            "I can't sleep and I want to kill myself", is_volatile=True
        )
        assert payload.is_crisis is True
        # Should NOT have matched sleep triggers (halted)
        assert len(payload.matched_triggers) == 0

    def test_normal_text_no_crisis(self, engine):
        payload = engine.recommend("I had a normal day at work")
        assert payload.is_crisis is False
        assert payload.crisis_message is None


# ---------------------------------------------------------------------------
# Layer 2: Context Matcher
# ---------------------------------------------------------------------------


class TestContextMatcher:
    def test_sleep_trigger(self, engine):
        payload = engine.recommend("I can't sleep at all")
        assert "sleep" in payload.matched_triggers
        assert any("sleep" in iv.title.lower() or "breath" in iv.title.lower()
                    for iv in payload.interventions)

    def test_money_trigger(self, engine):
        payload = engine.recommend("I'm worried about my debt and bills")
        assert "money" in payload.matched_triggers

    def test_exam_trigger(self, engine):
        payload = engine.recommend("My finals are next week and I'm panicking")
        assert "exam" in payload.matched_triggers

    def test_work_trigger(self, engine):
        payload = engine.recommend("My boss is terrible and I have a deadline")
        assert "work" in payload.matched_triggers

    def test_multiple_triggers(self, engine):
        payload = engine.recommend("I can't sleep because of work stress and debt")
        assert "sleep" in payload.matched_triggers
        assert "work" in payload.matched_triggers
        assert "money" in payload.matched_triggers

    def test_no_triggers_for_generic_text(self, engine):
        payload = engine.recommend("Today was a beautiful sunny day")
        assert len(payload.matched_triggers) == 0
        assert len(payload.interventions) == 0


# ---------------------------------------------------------------------------
# Layer 3: Preventive Nudges
# ---------------------------------------------------------------------------


class TestPreventiveNudges:
    def test_volatile_user_gets_nudges(self, engine):
        payload = engine.recommend(
            "Today was a beautiful day", is_volatile=True
        )
        assert len(payload.interventions) > 0
        assert any(iv.category == "grounding" for iv in payload.interventions)

    def test_non_volatile_no_nudges(self, engine):
        payload = engine.recommend("Today was a beautiful day", is_volatile=False)
        assert len(payload.interventions) == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestRecommendationIntegration:
    def test_interventions_sorted_by_priority(self, engine):
        payload = engine.recommend("I can't sleep and work is stressful")
        priorities = [iv.priority for iv in payload.interventions]
        assert priorities == sorted(priorities, reverse=True)

    def test_payload_dataclass_fields(self, engine):
        payload = engine.recommend("test text")
        assert isinstance(payload, RecommendationPayload)
        assert isinstance(payload.interventions, list)
        assert isinstance(payload.matched_triggers, list)
        assert isinstance(payload.is_crisis, bool)


# ---------------------------------------------------------------------------
# Layer 2: New trigger categories (relationship, health, grief, loneliness)
# ---------------------------------------------------------------------------


class TestNewTriggerCategories:
    def test_relationship_trigger(self, engine):
        payload = engine.recommend("My girlfriend and I had a terrible fight")
        assert "relationship" in payload.matched_triggers
        assert len(payload.interventions) > 0

    def test_relationship_breakup_trigger(self, engine):
        payload = engine.recommend("We just broke up and I am devastated")
        assert "relationship" in payload.matched_triggers

    def test_health_trigger(self, engine):
        payload = engine.recommend("I have been sick and in a lot of pain")
        assert "health" in payload.matched_triggers
        assert len(payload.interventions) > 0

    def test_grief_trigger(self, engine):
        payload = engine.recommend("I am grieving the loss of my father")
        assert "grief" in payload.matched_triggers
        assert len(payload.interventions) > 0

    def test_grief_passed_away(self, engine):
        payload = engine.recommend("My grandmother passed away last week")
        assert "grief" in payload.matched_triggers

    def test_loneliness_trigger(self, engine):
        payload = engine.recommend("I feel so lonely and isolated all the time")
        assert "loneliness" in payload.matched_triggers
        assert len(payload.interventions) > 0

    def test_loneliness_alone_trigger(self, engine):
        payload = engine.recommend("I have no friends and nobody cares about me")
        assert "loneliness" in payload.matched_triggers

    def test_new_categories_have_two_interventions_each(self, engine):
        for category, text in [
            ("relationship", "My partner and I keep arguing"),
            ("health", "I have been dealing with a chronic illness"),
            ("grief", "I am still grieving the loss"),
            ("loneliness", "I feel lonely and excluded"),
        ]:
            payload = engine.recommend(text)
            matched_ivs = [
                iv for iv in payload.interventions
                if category in payload.matched_triggers
            ]
            assert len(matched_ivs) >= 2, (
                f"Expected ≥2 interventions for '{category}', got {len(matched_ivs)}"
            )


# ---------------------------------------------------------------------------
# Layer 4: Escalation Tracker
# ---------------------------------------------------------------------------


class TestEscalationTracker:
    def test_requires_escalation_flag_set(self, engine):
        payload = engine.recommend(
            "I am still so stressed", requires_escalation=True
        )
        assert payload.requires_escalation is True

    def test_escalation_adds_high_priority_intervention(self, engine):
        payload = engine.recommend(
            "I am so stressed", requires_escalation=True
        )
        assert any(iv.priority >= 5 for iv in payload.interventions)

    def test_no_escalation_by_default(self, engine):
        payload = engine.recommend("I feel stressed at work")
        assert payload.requires_escalation is False

    def test_escalation_intervention_category_is_resource(self, engine):
        payload = engine.recommend(
            "Everything is overwhelming", requires_escalation=True
        )
        escalation_ivs = [iv for iv in payload.interventions if iv.priority >= 5]
        assert any(iv.category == "resource" for iv in escalation_ivs)

    def test_escalation_mentions_counsellor(self, engine):
        payload = engine.recommend(
            "I feel hopeless about everything", requires_escalation=True
        )
        text = " ".join(iv.description for iv in payload.interventions)
        assert any(
            kw in text.lower()
            for kw in ("counsellor", "counselor", "therapist", "professional", "samhsa")
        )

    def test_crisis_takes_priority_over_escalation(self, engine):
        """Crisis (Layer 1) must still halt processing even with escalation flag."""
        payload = engine.recommend(
            "I want to end my life", requires_escalation=True
        )
        assert payload.is_crisis is True
