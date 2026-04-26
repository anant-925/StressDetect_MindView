"""
intervention/engine.py
======================
Phase 3: Safety-First Recommendation Engine

Three-layer architecture ensuring user safety above all else:

Layer 1 — **Circuit Breaker** (regex)
    Matches crisis keywords (e.g. suicide, self-harm). If triggered,
    returns an emergency safety payload and HALTS all further AI processing.

Layer 2 — **Context Matcher** (keyword extractor)
    Lightweight keyword/regex extractor to identify common stress triggers
    (sleep, money, exam, work) and map them to specific micro-interventions.

Layer 3 — **Preventive Nudges**
    Accepts ``is_volatile`` (from temporal model). If ``True``, suggests
    grounding exercises even when the current score is low.

Design Guardrails
-----------------
- Layer 1 ALWAYS runs first and can short-circuit everything.
- Emergency payload includes the 988 Suicide & Crisis Lifeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Intervention:
    """A single recommended intervention."""

    title: str
    description: str
    category: str  # 'emergency', 'breathing', 'grounding', 'cognitive', 'resource'
    priority: int = 0  # higher = more urgent


@dataclass
class RecommendationPayload:
    """Full recommendation response."""

    is_crisis: bool = False
    crisis_message: Optional[str] = None
    interventions: list[Intervention] = field(default_factory=list)
    matched_triggers: list[str] = field(default_factory=list)
    requires_escalation: bool = False  # Layer 4: sustained high-stress flag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Layer 1: Crisis regex (case-insensitive, word-boundary)
_CRISIS_PATTERN = re.compile(
    r"\b(suicide|suicidal|kill\s+myself|kill\s+me|end\s+it\s+all"
    r"|end\s+my\s+life|want\s+to\s+die|wanna\s+die|don['\u2019]?t\s+want\s+to\s+live"
    r"|self[- ]?harm(?:ing)?|cut\s+myself)\b",
    re.IGNORECASE,
)

# Layer 2: Trigger → interventions mapping
_TRIGGER_INTERVENTIONS: dict[str, list[Intervention]] = {
    "sleep": [
        Intervention(
            title="4-7-8 Breathing Technique",
            description=(
                "Inhale for 4 seconds, hold for 7 seconds, exhale for "
                "8 seconds. Repeat 3-4 times. This activates the "
                "parasympathetic nervous system and promotes sleep."
            ),
            category="breathing",
            priority=3,
        ),
        Intervention(
            title="Progressive Muscle Relaxation",
            description=(
                "Starting from your toes, tense each muscle group for "
                "5 seconds, then release. Work up to your head."
            ),
            category="grounding",
            priority=2,
        ),
    ],
    "money": [
        Intervention(
            title="Financial Stress Grounding",
            description=(
                "Write down 3 financial facts you know for certain. "
                "Separate facts from fears. Consider calling 211 for "
                "local financial assistance resources."
            ),
            category="cognitive",
            priority=3,
        ),
        Intervention(
            title="5-4-3-2-1 Sensory Grounding",
            description=(
                "Name 5 things you see, 4 you hear, 3 you touch, "
                "2 you smell, 1 you taste. This interrupts the "
                "anxiety spiral."
            ),
            category="grounding",
            priority=2,
        ),
    ],
    "exam": [
        Intervention(
            title="Box Breathing",
            description=(
                "Breathe in for 4 counts, hold 4, out 4, hold 4. "
                "Repeat 4 cycles. Proven to reduce exam anxiety."
            ),
            category="breathing",
            priority=3,
        ),
        Intervention(
            title="Cognitive Reframe",
            description=(
                "Replace 'I will fail' with 'I have prepared and I "
                "will do my best.' Write down 3 things you've already "
                "studied well."
            ),
            category="cognitive",
            priority=2,
        ),
    ],
    "work": [
        Intervention(
            title="Micro-Break Protocol",
            description=(
                "Stand up, stretch for 60 seconds, look at something "
                "20 feet away for 20 seconds. Set a timer for the next "
                "break in 25 minutes (Pomodoro)."
            ),
            category="resource",
            priority=3,
        ),
        Intervention(
            title="Diaphragmatic Breathing",
            description=(
                "Place one hand on your chest, one on your belly. "
                "Breathe so only the belly hand moves. 6 breaths per "
                "minute for 2 minutes."
            ),
            category="breathing",
            priority=2,
        ),
    ],
    "relationship": [
        Intervention(
            title="Compassionate Communication",
            description=(
                "Write down what you need from the relationship right now "
                "without blame — focus on feelings and needs. "
                "If the conflict feels unsafe, contact the "
                "National Domestic Violence Hotline: 1-800-799-7233."
            ),
            category="cognitive",
            priority=3,
        ),
        Intervention(
            title="Self-Compassion Break",
            description=(
                "Place a hand on your heart and say: 'This is a moment of "
                "suffering. Suffering is part of life. May I be kind to "
                "myself.' Repeat slowly three times."
            ),
            category="grounding",
            priority=2,
        ),
    ],
    "health": [
        Intervention(
            title="Body-Scan Breathing",
            description=(
                "Close your eyes. Take a slow breath and bring attention "
                "to any area of discomfort. Breathe gently into it for "
                "3 cycles, then release. Repeat for each area."
            ),
            category="breathing",
            priority=3,
        ),
        Intervention(
            title="Reach Out to a Health Professional",
            description=(
                "Persistent health worry deserves professional attention. "
                "Consider speaking with your GP or calling a nurse helpline. "
                "In the US: 1-800-994-9662 (Women's Health Helpline) or "
                "your insurance nurse line."
            ),
            category="resource",
            priority=2,
        ),
    ],
    "grief": [
        Intervention(
            title="Grief Acknowledgement",
            description=(
                "Grief has no timeline. Allow yourself to feel without "
                "judgment. Write one sentence about what you miss most — "
                "expressing grief is not weakness, it is healing."
            ),
            category="cognitive",
            priority=3,
        ),
        Intervention(
            title="Grief Support Resources",
            description=(
                "You do not have to carry this alone. "
                "GriefShare: https://www.griefshare.org (support groups). "
                "National Alliance for Grieving Children: childrengrieve.org. "
                "Or simply call a trusted person right now."
            ),
            category="resource",
            priority=2,
        ),
    ],
    "loneliness": [
        Intervention(
            title="One Small Connection",
            description=(
                "Loneliness is reduced by even tiny moments of human "
                "connection. Send one message to someone you care about — "
                "it doesn't have to be deep. A 'thinking of you' is enough."
            ),
            category="cognitive",
            priority=3,
        ),
        Intervention(
            title="Loneliness Support Line",
            description=(
                "The Campaign to End Loneliness: "
                "https://www.campaigntoendloneliness.org/feeling-lonely. "
                "AARP Connect2Affect: connect2affect.org. "
                "Or try a local community event — even a brief outing helps."
            ),
            category="resource",
            priority=2,
        ),
    ],
}

# Layer 2: Trigger detection patterns (case-insensitive)
_TRIGGER_PATTERNS: dict[str, re.Pattern] = {
    "sleep": re.compile(
        r"\b(sleep|insomnia|can'?t\s+sleep|tired|exhausted|fatigue)\b",
        re.IGNORECASE,
    ),
    "money": re.compile(
        r"\b(money|financial|debt|bills?|rent|broke|bankrupt|loan)\b",
        re.IGNORECASE,
    ),
    "exam": re.compile(
        r"\b(exam|test|quiz|finals?|midterm|grade|gpa|study|studying)\b",
        re.IGNORECASE,
    ),
    "work": re.compile(
        r"\b(work|job|boss|coworker|deadline|fired|layoff|overtime|burnout)\b",
        re.IGNORECASE,
    ),
    "relationship": re.compile(
        r"\b(relationship|partner|girlfriend|boyfriend|spouse|husband|wife"
        r"|divorce|breakup|break.?up|broke.?up|cheating|argument|fight|conflict|dating)\b",
        re.IGNORECASE,
    ),
    "health": re.compile(
        r"\b(health|sick|illness|disease|pain|doctor|hospital|diagnosis"
        r"|medication|surgery|chronic|injury|symptoms?)\b",
        re.IGNORECASE,
    ),
    "grief": re.compile(
        r"\b(grief|griev(?:ing|ed)?|loss|lost|death|died|passed\s+away"
        r"|bereavement|mourning|miss(?:ing)?\s+(?:you|them|him|her))\b",
        re.IGNORECASE,
    ),
    "loneliness": re.compile(
        r"\b(lonely|loneliness|alone|isolated|isolation|no\s+friends?"
        r"|nobody|no\s+one|left\s+out|excluded|invisible)\b",
        re.IGNORECASE,
    ),
}

# Layer 4: Escalation intervention — shown when stress has been sustained
_ESCALATION_INTERVENTION = Intervention(
    title="Consider Speaking to a Professional",
    description=(
        "Your stress has remained elevated over several check-ins. "
        "Speaking with a counsellor, therapist, or trusted person can "
        "make a real difference. You deserve support.\n\n"
        "📞 SAMHSA Helpline: 1-800-662-4357 (free, confidential, 24/7)\n"
        "🌐 Find a therapist: https://www.psychologytoday.com/us/therapists"
    ),
    category="resource",
    priority=5,
)

# Layer 3: Preventive nudges for volatile users
_VOLATILE_NUDGES: list[Intervention] = [
    Intervention(
        title="Grounding Exercise: 5-4-3-2-1",
        description=(
            "Your recent stress patterns show variability. Take a moment: "
            "Name 5 things you see, 4 you hear, 3 you can touch, "
            "2 you smell, 1 you taste."
        ),
        category="grounding",
        priority=4,
    ),
    Intervention(
        title="Body Scan Check-In",
        description=(
            "Close your eyes. Scan from head to toe — notice any tension. "
            "Breathe into those areas. Even 60 seconds helps."
        ),
        category="grounding",
        priority=3,
    ),
]

# Emergency payload
_EMERGENCY_MESSAGE = (
    "⚠️ We noticed language that suggests you may be in crisis. "
    "You are not alone. Please reach out to a professional:\n\n"
    "📞 **988 Suicide & Crisis Lifeline** — Call or text **988** (US)\n"
    "📞 **Crisis Text Line** — Text **HOME** to **741741**\n"
    "🌐 **International Association for Suicide Prevention**: "
    "https://www.iasp.info/resources/Crisis_Centres/"
)


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------


class RecommendationEngine:
    """Three-layer safety-first recommendation engine.

    Usage
    -----
    >>> engine = RecommendationEngine()
    >>> payload = engine.recommend("I can't sleep at all", is_volatile=False)
    >>> for iv in payload.interventions:
    ...     print(iv.title)
    """

    def recommend(
        self,
        text: str,
        stress_score: float = 0.0,
        is_volatile: bool = False,
        requires_escalation: bool = False,
    ) -> RecommendationPayload:
        """Generate recommendations for the given text.

        Parameters
        ----------
        text : str
            The user's input text.
        stress_score : float
            The model's stress probability (0-1).
        is_volatile : bool
            Whether the user's temporal profile is volatile.
        requires_escalation : bool
            Whether the user has had 3+ consecutive above-threshold sessions.
            When ``True`` a high-priority counsellor-referral intervention is
            prepended to the payload.

        Returns
        -------
        RecommendationPayload
        """
        payload = RecommendationPayload()

        # ── Layer 1: Circuit Breaker ──
        if _CRISIS_PATTERN.search(text):
            payload.is_crisis = True
            payload.crisis_message = _EMERGENCY_MESSAGE
            payload.interventions.append(
                Intervention(
                    title="Immediate Crisis Support",
                    description=_EMERGENCY_MESSAGE,
                    category="emergency",
                    priority=10,
                )
            )
            # HALT — do not run further layers
            return payload

        # ── Layer 2: Context Matcher ──
        for trigger, pattern in _TRIGGER_PATTERNS.items():
            if pattern.search(text):
                payload.matched_triggers.append(trigger)
                payload.interventions.extend(
                    _TRIGGER_INTERVENTIONS[trigger]
                )

        # ── Layer 3: Preventive Nudges ──
        if is_volatile:
            payload.interventions.extend(_VOLATILE_NUDGES)

        # ── Layer 4: Escalation Tracker ──
        if requires_escalation:
            payload.requires_escalation = True
            payload.interventions.append(_ESCALATION_INTERVENTION)

        # Sort interventions by priority (highest first)
        payload.interventions.sort(key=lambda iv: iv.priority, reverse=True)

        return payload
