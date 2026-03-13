"""Dataset generator for CRI Benchmark — Contextual Resonance Index.

Generates synthetic benchmark datasets from :class:`RichPersonaSpec`
definitions. The generator is an **offline tool** — it does not call
external LLM APIs. Instead it uses deterministic, template-based
conversation synthesis seeded with a reproducible PRNG so that the same
seed always produces the same dataset.

Typical usage::

    from cri.models import GeneratorConfig
    from cri.datasets.generator import DatasetGenerator

    gen = DatasetGenerator(GeneratorConfig(seed=42))
    datasets = gen.generate_canonical_suite()

    for ds in datasets:
        gen.save_dataset(ds, Path("datasets/canonical") / ds.metadata.persona_id)

The resulting directory layout per dataset is::

    persona-1-basic/
    ├── conversations.jsonl   — one Message JSON object per line
    ├── ground_truth.json     — single GroundTruth JSON object
    └── metadata.json         — single DatasetMetadata JSON object

This format is consumed by :func:`cri.datasets.loader.load_dataset`.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cri.datasets.personas.specs import ALL_PERSONAS, RichPersonaSpec
from cri.models import (
    BeliefChange,
    ConflictScenario,
    ConversationDataset,
    DatasetMetadata,
    ForgettableFact,
    GeneratorConfig,
    GroundTruth,
    Message,
    NoiseExample,
    ProfileDimension,
    SignalExample,
    TemporalFact,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation templates
# ---------------------------------------------------------------------------

_USER_GREETINGS = [
    "Hey, how's it going?",
    "Hi there!",
    "Good morning!",
    "Hey!",
    "Hi, hope you're doing well.",
    "Hello!",
]

_ASSISTANT_GREETINGS = [
    "Hi! How can I help you today?",
    "Hey there! What's on your mind?",
    "Hello! Great to hear from you.",
    "Hi! Hope you're having a good day. What can I do for you?",
    "Hey! What's up?",
]

_ASSISTANT_ACKNOWLEDGMENTS = [
    "That's really interesting, tell me more!",
    "Thanks for sharing that with me.",
    "Oh wow, that sounds great!",
    "I appreciate you telling me about that.",
    "That's good to know!",
    "Interesting! How's that going for you?",
    "Got it, thanks for the update.",
    "That makes a lot of sense.",
    "Sounds like things are going well!",
    "I'll keep that in mind.",
]

_ASSISTANT_FOLLOWUPS = [
    "How has that been going for you?",
    "What made you decide to make that change?",
    "That's a big step! How are you feeling about it?",
    "Do you want to talk more about that?",
    "Is there anything I can help with regarding that?",
]

_USER_CASUAL = [
    "Just checking in, nothing specific today.",
    "Having a quiet day, just wanted to chat.",
    "Can you tell me something interesting?",
    "What do you think about the news lately?",
    "Any recommendations for a good book?",
    "I'm bored, entertain me.",
    "Random thought — do you think AI will ever be truly creative?",
    "What's a fun fact I probably don't know?",
]

_SIGNAL_TEMPLATES_OCCUPATION = [
    "I work as a {value}. It keeps me busy but I love it.",
    "My job as a {value} is really rewarding.",
    "Being a {value} means long hours, but it's worth it.",
    "I've been working as a {value} for a while now.",
]

_SIGNAL_TEMPLATES_LOCATION = [
    "I live in {value}. It's a great city.",
    "I'm based in {value} right now.",
    "{value} has been home for a while now.",
    "Living in {value} has its perks — the food, the culture.",
]

_SIGNAL_TEMPLATES_HOBBY = [
    "I've been really into {value} lately.",
    "One of my favorite things to do is {value}.",
    "{value} is how I unwind after a long week.",
    "I spent the whole weekend doing {value}. So relaxing.",
]

_SIGNAL_TEMPLATES_GENERIC = [
    "By the way, my {dim} is {value}.",
    "Did I ever mention? My {dim} is {value}.",
    "Just so you know, regarding {dim} — it's {value}.",
    "Speaking of {dim}, mine is {value}.",
]

_BELIEF_CHANGE_TEMPLATES = [
    "Actually, I need to update you — I used to {old}, but now I {new}.",
    "Big change: I'm no longer into {old}. I've switched to {new}.",
    "Things have changed — instead of {old}, I'm now doing {new}.",
    "I've moved on from {old}. These days it's all about {new}.",
]


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------


class DatasetGenerator:
    """Generates synthetic CRI benchmark datasets from persona specifications.

    The generator is deterministic: given the same :class:`GeneratorConfig`
    (specifically the same ``seed``), it will always produce identical
    datasets. This ensures benchmark reproducibility.

    Args:
        config: Generator configuration including seed, LLM model, and
                generation parameters.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        logger.info(
            "DatasetGenerator initialised (seed=%s, model=%s)",
            config.seed,
            config.llm_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, persona: RichPersonaSpec) -> ConversationDataset:
        """Generate a complete conversation dataset for a persona.

        Creates a realistic multi-session conversation that weaves in
        signal messages (facts about the persona), noise messages (casual
        chat), belief changes, conflict statements, and temporal fact
        references at appropriate points throughout the timeline.

        Args:
            persona: Rich persona specification with ground-truth data.

        Returns:
            A fully populated :class:`ConversationDataset` ready for
            evaluation or serialisation.
        """
        logger.info(
            "Generating dataset for persona '%s' (%s) — target %d msgs over %d days",
            persona.name,
            persona.persona_id,
            persona.target_message_count,
            persona.simulated_days,
        )

        # 1. Plan sessions across the simulated days
        sessions = self._plan_sessions(persona)
        logger.debug("Planned %d sessions across %d days", len(sessions), persona.simulated_days)

        # 2. Build the event schedule — what content goes into which session
        schedule = self._build_schedule(persona, sessions)

        # 3. Generate messages from the schedule
        messages = self._generate_messages(persona, sessions, schedule)
        logger.info(
            "Generated %d messages for '%s' (target was %d)",
            len(messages),
            persona.name,
            persona.target_message_count,
        )

        # 4. Build ground truth from the persona spec
        ground_truth = self._build_ground_truth(persona, messages)

        # 5. Build metadata
        metadata = DatasetMetadata(
            dataset_id=persona.persona_id,
            persona_id=persona.persona_id,
            message_count=len(messages),
            simulated_days=persona.simulated_days,
            version="1.0.0",
            seed=self.config.seed,
        )

        dataset = ConversationDataset(
            metadata=metadata,
            messages=messages,
            ground_truth=ground_truth,
        )

        logger.info(
            "Dataset for '%s' complete: %d messages, %d profile dims, "
            "%d belief changes, %d conflicts",
            persona.name,
            len(messages),
            len(ground_truth.final_profile),
            len(ground_truth.changes),
            len(ground_truth.conflicts),
        )
        return dataset

    def generate_canonical_suite(self) -> list[ConversationDataset]:
        """Generate datasets for all canonical personas.

        Returns:
            A list of :class:`ConversationDataset` instances, one per
            canonical persona (basic, intermediate, advanced).
        """
        logger.info("Generating canonical suite (%d personas)", len(ALL_PERSONAS))
        datasets: list[ConversationDataset] = []
        for persona in ALL_PERSONAS:
            ds = self.generate(persona)
            datasets.append(ds)
        logger.info("Canonical suite complete: %d datasets", len(datasets))
        return datasets

    def save_dataset(
        self,
        dataset: ConversationDataset,
        output_dir: Path,
    ) -> None:
        """Persist a dataset to disk in the canonical directory layout.

        Creates three files:

        - ``conversations.jsonl`` — one :class:`Message` JSON per line
        - ``ground_truth.json`` — single :class:`GroundTruth` JSON object
        - ``metadata.json`` — single :class:`DatasetMetadata` JSON object

        Args:
            dataset: The dataset to save.
            output_dir: Target directory (created if it does not exist).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # conversations.jsonl
        conversations_path = output_dir / "conversations.jsonl"
        with open(conversations_path, "w", encoding="utf-8") as fh:
            for msg in dataset.messages:
                fh.write(msg.model_dump_json() + "\n")
        logger.info("Wrote %d messages to %s", len(dataset.messages), conversations_path)

        # ground_truth.json
        gt_path = output_dir / "ground_truth.json"
        gt_path.write_text(
            dataset.ground_truth.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote ground truth to %s", gt_path)

        # metadata.json
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(
            dataset.metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote metadata to %s", meta_path)

    # ------------------------------------------------------------------
    # Session planning
    # ------------------------------------------------------------------

    def _plan_sessions(
        self,
        persona: RichPersonaSpec,
    ) -> list[dict[str, Any]]:
        """Plan conversation sessions across the simulated timeline.

        Creates 2-5 sessions per week with varying lengths, ensuring the
        total message count approximates the persona's target.

        Returns:
            Ordered list of session descriptors with keys:
            ``session_id``, ``day``, ``date``, ``target_msgs``.
        """
        base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        days = persona.simulated_days
        target_msgs = persona.target_message_count

        # Decide which days have sessions (2-5 per 7-day week)
        session_days: list[int] = []
        for week_start in range(0, days, 7):
            week_end = min(week_start + 7, days)
            available = list(range(week_start, week_end))
            n_sessions = min(
                len(available),
                self._rng.randint(2, 5),
            )
            chosen = sorted(self._rng.sample(available, n_sessions))
            session_days.extend(chosen)

        if not session_days:
            session_days = [0]

        # Distribute messages across sessions
        n_sessions = len(session_days)
        base_per_session = target_msgs // n_sessions
        remainder = target_msgs % n_sessions

        sessions: list[dict[str, Any]] = []
        for idx, day in enumerate(session_days):
            msgs = base_per_session + (1 if idx < remainder else 0)
            # Add some variance (±30%)
            variance = self._rng.uniform(0.7, 1.3)
            msgs = max(4, int(msgs * variance))  # minimum 4 messages per session
            # Ensure even number (user-assistant pairs)
            if msgs % 2 != 0:
                msgs += 1
            sessions.append(
                {
                    "session_id": f"session-{idx + 1:03d}",
                    "day": day + 1,  # 1-indexed
                    "date": base_date + timedelta(days=day),
                    "target_msgs": msgs,
                }
            )

        logger.debug(
            "Planned %d sessions, total target ≈ %d msgs",
            len(sessions),
            sum(s["target_msgs"] for s in sessions),
        )
        return sessions

    # ------------------------------------------------------------------
    # Schedule building
    # ------------------------------------------------------------------

    def _build_schedule(
        self,
        persona: RichPersonaSpec,
        sessions: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        """Assign content items (signals, noise, changes, conflicts) to sessions.

        Returns:
            A dict mapping session index → list of content items. Each item
            is a dict with ``type`` ('signal', 'noise', 'change', 'conflict',
            'profile', 'temporal', 'casual') and relevant payload.
        """
        schedule: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(sessions))}

        n_sessions = len(sessions)

        # Map message-index positions to session indices
        def msg_to_session(msg_idx: int) -> int:
            """Approximate which session a global message index falls into."""
            cumulative = 0
            for si, sess in enumerate(sessions):
                cumulative += sess["target_msgs"]
                if msg_idx <= cumulative:
                    return si
            return n_sessions - 1

        # Distribute profile dimension signals across early sessions
        dim_items = list(persona.profile_dimensions.items())
        self._rng.shuffle(dim_items)
        early_sessions = max(1, n_sessions // 3)
        for idx, (dim_name, dim) in enumerate(dim_items):
            target_session = idx % early_sessions
            schedule[target_session].append(
                {
                    "type": "profile",
                    "dimension_name": dim_name,
                    "dimension": dim,
                }
            )

        # Place belief changes at their approximate message positions
        for change in persona.belief_changes:
            si = msg_to_session(change.changed_around_msg)
            schedule[si].append(
                {
                    "type": "change",
                    "change": change,
                }
            )

        # Place conflict statements
        for conflict in persona.conflicts:
            for msg_idx in conflict.introduced_at_messages:
                si = msg_to_session(msg_idx)
                schedule[si].append(
                    {
                        "type": "conflict",
                        "conflict": conflict,
                        "statement_idx": conflict.introduced_at_messages.index(msg_idx),
                    }
                )

        # Distribute signal examples
        for signal in persona.signal_examples:
            si = self._rng.randint(0, n_sessions - 1)
            schedule[si].append(
                {
                    "type": "signal",
                    "signal": signal,
                }
            )

        # Distribute noise examples
        for noise in persona.noise_examples:
            si = self._rng.randint(0, n_sessions - 1)
            schedule[si].append(
                {
                    "type": "noise",
                    "noise": noise,
                }
            )

        # Distribute temporal fact references
        for tf in persona.temporal_facts:
            si = self._rng.randint(0, n_sessions - 1)
            schedule[si].append(
                {
                    "type": "temporal",
                    "temporal_fact": tf,
                }
            )

        # Distribute forgettable fact mentions (early in the conversation)
        for ff in persona.forgettable_facts:
            si = msg_to_session(ff.mentioned_at_message)
            schedule[si].append(
                {
                    "type": "forgettable",
                    "forgettable_fact": ff,
                }
            )

        return schedule

    # ------------------------------------------------------------------
    # Message generation
    # ------------------------------------------------------------------

    def _generate_messages(
        self,
        persona: RichPersonaSpec,
        sessions: list[dict[str, Any]],
        schedule: dict[int, list[dict[str, Any]]],
    ) -> list[Message]:
        """Generate all messages across all sessions.

        Returns:
            Ordered list of :class:`Message` objects with sequential IDs.
        """
        all_messages: list[Message] = []
        message_id = 1

        for si, session in enumerate(sessions):
            session_msgs = self._generate_session_messages(
                persona=persona,
                session=session,
                items=schedule.get(si, []),
                start_id=message_id,
            )
            all_messages.extend(session_msgs)
            message_id += len(session_msgs)

            if (si + 1) % 10 == 0:
                logger.debug(
                    "Generated sessions %d/%d (%d messages so far)",
                    si + 1,
                    len(sessions),
                    len(all_messages),
                )

        return all_messages

    def _generate_session_messages(
        self,
        persona: RichPersonaSpec,
        session: dict[str, Any],
        items: list[dict[str, Any]],
        start_id: int,
    ) -> list[Message]:
        """Generate messages for a single session.

        Produces alternating user/assistant message pairs. Scheduled
        content items are woven into the conversation at random positions.
        """
        target = session["target_msgs"]
        session_id = session["session_id"]
        day = session["day"]
        base_time: datetime = session["date"]

        # How many user-assistant pairs
        n_pairs = target // 2

        # Build a list of user messages to create
        user_contents: list[str] = []

        # Add scheduled content items as user messages
        for item in items:
            content = self._item_to_user_message(persona, item)
            if content:
                user_contents.append(content)

        # Fill remaining slots with casual / filler messages
        while len(user_contents) < n_pairs:
            user_contents.append(self._generate_filler_user_message(persona))

        # Shuffle to distribute scheduled items throughout the session
        self._rng.shuffle(user_contents)

        # Trim if we have too many
        user_contents = user_contents[:n_pairs]

        # Generate message pairs
        messages: list[Message] = []
        for i, user_text in enumerate(user_contents):
            # Timestamp: spread messages across the day (9am-10pm)
            minutes_offset = int((i / max(n_pairs, 1)) * 780)  # 13h * 60min
            ts = base_time.replace(hour=9, minute=0, second=0) + timedelta(
                minutes=minutes_offset,
                seconds=self._rng.randint(0, 59),
            )
            ts_str = ts.isoformat()

            # User message
            messages.append(
                Message(
                    message_id=start_id + len(messages),
                    role="user",
                    content=user_text,
                    timestamp=ts_str,
                    session_id=session_id,
                    day=day,
                )
            )

            # Assistant response
            assistant_text = self._generate_assistant_response(user_text)
            ts_reply = ts + timedelta(seconds=self._rng.randint(1, 30))

            messages.append(
                Message(
                    message_id=start_id + len(messages),
                    role="assistant",
                    content=assistant_text,
                    timestamp=ts_reply.isoformat(),
                    session_id=session_id,
                    day=day,
                )
            )

        return messages

    # ------------------------------------------------------------------
    # Content → message conversion
    # ------------------------------------------------------------------

    def _item_to_user_message(
        self,
        persona: RichPersonaSpec,
        item: dict[str, Any],
    ) -> str | None:
        """Convert a scheduled content item to a user message string."""
        item_type = item["type"]

        if item_type == "profile":
            return self._profile_dim_to_message(
                item["dimension_name"],
                item["dimension"],
            )

        if item_type == "signal":
            signal: SignalExample = item["signal"]
            return signal.text

        if item_type == "noise":
            noise: NoiseExample = item["noise"]
            return noise.text

        if item_type == "change":
            change: BeliefChange = item["change"]
            return self._belief_change_to_message(change)

        if item_type == "conflict":
            conflict: ConflictScenario = item["conflict"]
            stmt_idx: int = item["statement_idx"]
            if stmt_idx < len(conflict.conflicting_statements):
                return conflict.conflicting_statements[stmt_idx]
            return None

        if item_type == "temporal":
            tf: TemporalFact = item["temporal_fact"]
            return self._temporal_fact_to_message(tf)

        if item_type == "forgettable":
            ff: ForgettableFact = item["forgettable_fact"]
            return ff.text

        return None

    def _profile_dim_to_message(
        self,
        dim_name: str,
        dim: ProfileDimension,
    ) -> str:
        """Generate a natural user message that reveals a profile dimension."""
        value = dim.value
        if isinstance(value, list):
            value_str = ", ".join(value[:-1]) + f" and {value[-1]}" if len(value) > 1 else value[0]
        else:
            value_str = str(value)

        templates = _SIGNAL_TEMPLATES_GENERIC
        if dim_name in ("occupation", "job", "career", "previous_occupation"):
            templates = _SIGNAL_TEMPLATES_OCCUPATION
        elif dim_name in ("location", "city", "residence"):
            templates = _SIGNAL_TEMPLATES_LOCATION
        elif dim_name in ("hobbies", "hobby", "interests"):
            templates = _SIGNAL_TEMPLATES_HOBBY

        template = self._rng.choice(templates)

        try:
            return template.format(
                value=value_str,
                dim=dim_name.replace("_", " "),
            )
        except (KeyError, IndexError):
            return f"My {dim_name.replace('_', ' ')} is {value_str}."

    def _belief_change_to_message(self, change: BeliefChange) -> str:
        """Generate a user message announcing a belief change."""
        template = self._rng.choice(_BELIEF_CHANGE_TEMPLATES)
        return template.format(
            old=change.old_value.lower(),
            new=change.new_value.lower(),
        )

    def _temporal_fact_to_message(self, tf: TemporalFact) -> str:
        """Generate a user message referencing a temporal fact."""
        if tf.should_be_current:
            return f"Currently, {tf.description.lower()} — {tf.value}."
        if tf.valid_until:
            return f"Back when {tf.description.lower()} was {tf.value}, things were different."
        return f"Just so you know, {tf.description.lower()}: {tf.value}."

    def _generate_filler_user_message(self, persona: RichPersonaSpec) -> str:
        """Generate a casual filler message for the user."""
        roll = self._rng.random()
        if roll < 0.3:
            return self._rng.choice(_USER_GREETINGS)
        if roll < 0.6:
            return self._rng.choice(_USER_CASUAL)
        # Generate a persona-flavoured casual message
        flavours = [
            f"Had a good day today. {persona.name.split()[0]} out.",
            "Just thinking out loud here...",
            "Nothing specific, just wanted to chat for a bit.",
            "How are things on your end?",
            "Any thoughts on what I should have for dinner?",
            "I've been meaning to ask — what's your favorite color?",
            "Do you ever get tired of answering questions?",
            "Quick question — never mind, I forgot what I was going to ask.",
            "Let me know if you need anything from me.",
            "I should probably be working but here I am chatting with you.",
        ]
        return self._rng.choice(flavours)

    def _generate_assistant_response(self, user_text: str) -> str:
        """Generate a plausible assistant response to a user message."""
        # Greeting detection
        greetings = {"hi", "hey", "hello", "good morning", "good afternoon", "good evening"}
        if user_text.lower().strip().rstrip("!.,") in greetings:
            return self._rng.choice(_ASSISTANT_GREETINGS)

        # If the message contains personal info, acknowledge + follow up
        personal_keywords = [
            "i work",
            "i live",
            "i've been",
            "i moved",
            "i decided",
            "i used to",
            "big change",
            "big news",
            "i just",
            "my job",
            "i'm a",
            "i am a",
            "i switched",
            "i've decided",
            "actually",
            "update",
            "change",
        ]
        lower = user_text.lower()
        if any(kw in lower for kw in personal_keywords):
            ack = self._rng.choice(_ASSISTANT_ACKNOWLEDGMENTS)
            if self._rng.random() < 0.5:
                followup = self._rng.choice(_ASSISTANT_FOLLOWUPS)
                return f"{ack} {followup}"
            return ack

        # Generic responses
        generic = [
            "Sure thing! Let me know if there's anything else.",
            "Of course. What else is on your mind?",
            "Happy to help! Anything else?",
            "No problem at all.",
            "Sounds good to me!",
            "I understand. Feel free to ask me anything.",
            "Great question! Let me think about that.",
            "That's a good point. Thanks for sharing.",
        ]
        return self._rng.choice(generic)

    # ------------------------------------------------------------------
    # Ground truth assembly
    # ------------------------------------------------------------------

    def _build_ground_truth(
        self,
        persona: RichPersonaSpec,
        messages: list[Message],
    ) -> GroundTruth:
        """Assemble the ground truth from the persona spec.

        Adjusts message references in belief changes and conflict scenarios
        to point to actual generated message IDs where possible.
        """
        msg_count = len(messages)

        # Remap belief change message references to be within bounds
        adjusted_changes: list[BeliefChange] = []
        for change in persona.belief_changes:
            clamped_around = min(change.changed_around_msg, msg_count)
            clamped_around = max(1, clamped_around)
            clamped_keys = [max(1, min(k, msg_count)) for k in change.key_messages]
            adjusted_changes.append(
                BeliefChange(
                    fact=change.fact,
                    old_value=change.old_value,
                    new_value=change.new_value,
                    query_topic=change.query_topic,
                    changed_around_msg=clamped_around,
                    key_messages=clamped_keys,
                )
            )

        # Remap conflict message references
        adjusted_conflicts: list[ConflictScenario] = []
        for conflict in persona.conflicts:
            clamped_msgs = [max(1, min(m, msg_count)) for m in conflict.introduced_at_messages]
            adjusted_conflicts.append(
                ConflictScenario(
                    conflict_id=conflict.conflict_id,
                    topic=conflict.topic,
                    conflicting_statements=conflict.conflicting_statements,
                    correct_resolution=conflict.correct_resolution,
                    resolution_type=conflict.resolution_type,
                    introduced_at_messages=clamped_msgs,
                )
            )

        # Clamp forgettable fact message references
        adjusted_forgettable: list[ForgettableFact] = []
        for ff in persona.forgettable_facts:
            adjusted_forgettable.append(
                ForgettableFact(
                    fact_id=ff.fact_id,
                    text=ff.text,
                    reason=ff.reason,
                    mentioned_at_message=max(1, min(ff.mentioned_at_message, msg_count)),
                    should_be_absent_after=max(1, min(ff.should_be_absent_after, msg_count)),
                )
            )

        return GroundTruth(
            final_profile=dict(persona.profile_dimensions),
            changes=adjusted_changes,
            noise_examples=list(persona.noise_examples),
            signal_examples=list(persona.signal_examples),
            conflicts=adjusted_conflicts,
            temporal_facts=list(persona.temporal_facts),
            query_relevance_pairs=list(persona.query_relevance_pairs),
            forgettable_facts=adjusted_forgettable,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DatasetGenerator",
]
