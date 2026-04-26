"""
utils/reward.py
===============
Reward computation for the RL-style feedback loop.

The reward signal bridges raw user/LLM feedback into a scalar weight used
during model retraining (see ``training/retrain.py``).

Design
------
- User feedback of ``1`` (prediction was correct) → positive reward (+1.0).
- User feedback of ``0`` (prediction was wrong)   → negative reward (-1.0).
- When an LLM judge is also available the two signals are averaged.
- The returned value is later used as a *loss weight* in ``weighted_loss``.
  Because loss weights must be non-negative, callers should use
  ``abs(reward)`` for weighting; the *sign* is only meaningful for
  logging / analytics.
"""

from __future__ import annotations


def compute_reward(user_feedback: int) -> float:
    """Compute a ±1 reward scalar from a user's binary feedback.

    Parameters
    ----------
    user_feedback : int
        ``1`` if the user confirmed the prediction was correct,
        ``0`` if they said it was wrong.

    Returns
    -------
    float
        ``+1.0`` for confirmed-correct, ``-1.0`` for confirmed-wrong.
    """
    return 1.0 if user_feedback == 1 else -1.0


def compute_combined_reward(
    user_feedback: int,
    llm_reward: int | None = None,
) -> float:
    """Combine user feedback with an optional LLM reward.

    Parameters
    ----------
    user_feedback : int
        1 = correct, 0 = wrong.
    llm_reward : int | None
        +1 or -1 from an LLM judge, or ``None`` when unavailable.

    Returns
    -------
    float
        The average of the user reward and LLM reward when both are
        present, otherwise just the user reward.
    """
    user_r = compute_reward(user_feedback)
    if llm_reward is not None:
        return (user_r + float(llm_reward)) / 2.0
    return user_r


def reward_to_weight(reward: float) -> float:
    """Convert a signed reward to a non-negative loss weight.

    Both positive and negative rewards provide a strong training signal;
    we therefore scale both to ``1.5`` so that feedback-derived samples
    are weighted 50 % higher than the default loss of ``1.0``.

    Parameters
    ----------
    reward : float
        Signed reward value from ``compute_reward`` / ``compute_combined_reward``.

    Returns
    -------
    float
        ``1.5`` for any non-zero reward, ``1.0`` for reward == 0.
    """
    return 1.5 if reward != 0.0 else 1.0
