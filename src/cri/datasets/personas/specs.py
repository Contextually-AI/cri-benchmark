"""Pre-defined persona specifications for canonical CRI datasets.

These personas represent different levels of complexity for benchmarking
memory systems. Each persona includes:

- Profile dimensions (ground truth facts)
- Belief changes (knowledge updates over time)
- Noise examples (irrelevant chatter)
- Signal examples (fact-bearing messages)
- Conflict scenarios (contradictory information)
- Temporal facts (time-bounded knowledge)
- Query-relevance pairs (retrieval precision tests)
- Forgettable facts (ephemeral/superseded information for SFC evaluation)

Complexity levels:

- **Basic**: Simple persona with clear attributes and a few changes.
- **Intermediate**: Evolving persona with career/location/preference changes.
- **Advanced**: Complex persona with contradictions, temporal gaps, opinion
  reversals, and nuanced belief evolution.

Profile dimension categories follow the 6-W framework:
- **WHO**: name, age, family, nationality
- **WHAT**: occupation, hobbies, diet, health
- **WHERE**: city, neighborhood, commute
- **WHEN**: schedule, routines, sleep patterns
- **WHY**: motivations, values, goals
- **HOW**: communication style, tech preferences
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cri.models import (
    BeliefChange,
    ConflictScenario,
    ForgettableFact,
    NoiseExample,
    ProfileDimension,
    QueryRelevancePair,
    SignalExample,
    TemporalFact,
)

# ---------------------------------------------------------------------------
# Rich Persona Spec
# ---------------------------------------------------------------------------


class RichPersonaSpec(BaseModel):
    """Full specification of a persona used to generate benchmark datasets.

    This model carries
    all ground-truth components needed to synthesise a complete
    :class:`~cri.models.ConversationDataset` — including belief changes,
    conflict scenarios, temporal facts, and query-relevance pairs.
    """

    persona_id: str = Field(
        description="Unique identifier for this persona (e.g. 'persona-1-basic')"
    )
    name: str = Field(description="Human-readable persona name")
    description: str = Field(description="Brief narrative description of the persona's background")
    complexity_level: Literal["basic", "intermediate", "advanced"] = Field(
        description="Benchmark complexity tier"
    )

    # Ground-truth components
    profile_dimensions: dict[str, ProfileDimension] = Field(
        description="Expected final profile keyed by dimension name"
    )
    belief_changes: list[BeliefChange] = Field(
        default_factory=list,
        description="Ordered list of belief changes that occur during the conversation",
    )
    noise_examples: list[NoiseExample] = Field(
        default_factory=list,
        description="Template noise messages to weave into the conversation",
    )
    signal_examples: list[SignalExample] = Field(
        default_factory=list,
        description="Template signal messages that reveal persona facts",
    )
    conflicts: list[ConflictScenario] = Field(
        default_factory=list,
        description="Conflict scenarios to embed in the conversation",
    )
    temporal_facts: list[TemporalFact] = Field(
        default_factory=list,
        description="Facts with temporal validity constraints",
    )
    query_relevance_pairs: list[QueryRelevancePair] = Field(
        default_factory=list,
        description="Query-relevance pairs for QRP evaluation",
    )
    forgettable_facts: list[ForgettableFact] = Field(
        default_factory=list,
        description="Facts that should be forgotten/discarded by end of conversation",
    )

    # Generation parameters
    simulated_days: int = Field(
        default=90,
        description="Number of simulated days the conversation should span",
    )
    target_message_count: int = Field(
        default=200,
        description="Approximate number of messages to generate",
    )


# ---------------------------------------------------------------------------
# Canonical Persona: Basic — Alex Chen
# ---------------------------------------------------------------------------

PERSONA_BASIC = RichPersonaSpec(
    persona_id="persona-1-basic",
    name="Alex Chen",
    description=(
        "A 28-year-old data analyst at a fintech startup in Denver, Colorado. "
        "He recently moved from San Francisco, has a rescue cat named Luna, "
        "enjoys rock climbing and board games, and recently went vegetarian. "
        "This persona tests core profile accuracy and simple belief changes."
    ),
    complexity_level="basic",
    simulated_days=30,
    target_message_count=1000,
    # 10 profile dimensions covering WHO/WHAT/WHERE/WHEN/WHY/HOW
    profile_dimensions={
        # WHO
        "name": ProfileDimension(
            dimension_name="name",
            value="Alex Chen",
            query_topic="name",
            category="who",
        ),
        "age": ProfileDimension(
            dimension_name="age",
            value="28",
            query_topic="age",
            category="who",
        ),
        "family_status": ProfileDimension(
            dimension_name="family_status",
            value="single",
            query_topic="relationship status",
            category="who",
        ),
        # WHAT
        "occupation": ProfileDimension(
            dimension_name="occupation",
            value="Data Analyst at a fintech startup",
            query_topic="occupation",
            category="what",
        ),
        "hobbies": ProfileDimension(
            dimension_name="hobbies",
            value=["rock climbing", "board games", "cooking Thai food"],
            query_topic="hobbies",
            category="what",
        ),
        "diet": ProfileDimension(
            dimension_name="diet",
            value="vegetarian",
            query_topic="dietary preference",
            category="what",
        ),
        # WHERE
        "city": ProfileDimension(
            dimension_name="city",
            value="Denver, Colorado",
            query_topic="current city",
            category="where",
        ),
        # WHEN
        "schedule": ProfileDimension(
            dimension_name="schedule",
            value="exercises in the mornings before work",
            query_topic="daily schedule",
            category="when",
        ),
        # WHY
        "goals": ProfileDimension(
            dimension_name="goals",
            value="learning Spanish on Duolingo, considering getting a second cat",
            query_topic="current goals",
            category="why",
        ),
        # HOW
        "communication_style": ProfileDimension(
            dimension_name="communication_style",
            value="casual and friendly, uses emojis occasionally",
            query_topic="communication style",
            category="how",
        ),
    },
    # 3 belief changes
    belief_changes=[
        BeliefChange(
            fact="city",
            old_value="San Francisco, California",
            new_value="Denver, Colorado",
            query_topic="current city",
            changed_around_msg=200,
            key_messages=[180, 195, 210],
        ),
        BeliefChange(
            fact="diet",
            old_value="omnivore",
            new_value="vegetarian",
            query_topic="dietary preference",
            changed_around_msg=500,
            key_messages=[480, 495, 510],
        ),
        BeliefChange(
            fact="pet_count",
            old_value="one cat (Luna)",
            new_value="considering getting a second cat",
            query_topic="pets",
            changed_around_msg=800,
            key_messages=[785, 800, 815],
        ),
    ],
    # 10 noise examples
    noise_examples=[
        NoiseExample(
            text="Hey, how's it going today?",
            reason="Generic greeting with no factual content about the user",
        ),
        NoiseExample(
            text="What's the weather going to be like this weekend?",
            reason="Weather inquiry unrelated to persona attributes",
        ),
        NoiseExample(
            text="Thanks for the help, that was really useful!",
            reason="Politeness expression with no persona information",
        ),
        NoiseExample(
            text="Can you help me write an email to my boss?",
            reason="Task request that doesn't reveal persona information",
        ),
        NoiseExample(
            text="Hmm, let me think about that for a second.",
            reason="Thinking-out-loud with no factual content",
        ),
        NoiseExample(
            text="LOL that's hilarious 😂",
            reason="Reaction with no informational content",
        ),
        NoiseExample(
            text="What do you think about the new ChatGPT update?",
            reason="General tech discussion unrelated to persona",
        ),
        NoiseExample(
            text="Remind me to call my mom later.",
            reason="Reminder request with minimal persona data",
        ),
        NoiseExample(
            text="Sorry, I got distracted. What were we talking about?",
            reason="Conversational filler with no factual content",
        ),
        NoiseExample(
            text="That makes sense, thanks for explaining!",
            reason="Acknowledgment with no persona information",
        ),
    ],
    # 10 signal examples
    signal_examples=[
        SignalExample(
            text="I work as a data analyst at a fintech startup here in Denver.",
            target_fact="occupation: Data Analyst, city: Denver",
        ),
        SignalExample(
            text=(
                "Luna, my rescue cat, is curled up on my desk again. "
                "She always does this when I'm working."
            ),
            target_fact="pet: rescue cat named Luna",
        ),
        SignalExample(
            text="Just got back from the climbing gym — managed to send a V6 today!",
            target_fact="hobbies: rock climbing",
        ),
        SignalExample(
            text="We had a board game night last Friday. Played Settlers of Catan and Wingspan.",
            target_fact="hobbies: board games",
        ),
        SignalExample(
            text="I made pad thai from scratch last night. Thai food is my comfort food.",
            target_fact="hobbies: cooking Thai food",
        ),
        SignalExample(
            text="I went vegetarian about six months ago and honestly I feel so much better.",
            target_fact="diet: vegetarian",
        ),
        SignalExample(
            text=(
                "I just moved to Denver from San Francisco. "
                "The cost of living difference is insane."
            ),
            target_fact="city change: San Francisco → Denver",
        ),
        SignalExample(
            text="I've been doing Duolingo for Spanish every day. 45 day streak!",
            target_fact="goals: learning Spanish on Duolingo",
        ),
        SignalExample(
            text="I always work out in the morning before heading to the office. Helps me focus.",
            target_fact="schedule: exercises in the mornings before work",
        ),
        SignalExample(
            text="I'm 28 and still figuring things out. But I love my job at least.",
            target_fact="age: 28",
        ),
    ],
    # 3 conflicts
    conflicts=[
        ConflictScenario(
            conflict_id="conflict-basic-01",
            topic="location",
            conflicting_statements=[
                "I live in San Francisco, love the Bay Area vibe.",
                "Denver is home now. I moved here a few months ago and I'm loving it.",
            ],
            correct_resolution=(
                "Alex moved from San Francisco to Denver. Denver is his current city."
            ),
            resolution_type="recency",
            introduced_at_messages=[50, 250],
        ),
        ConflictScenario(
            conflict_id="conflict-basic-02",
            topic="diet",
            conflicting_statements=[
                "I had the best burger at this place downtown yesterday.",
                "I've been vegetarian for six months now. No meat at all.",
            ],
            correct_resolution=(
                "Alex transitioned to vegetarian. The burger reference was before the change."
            ),
            resolution_type="recency",
            introduced_at_messages=[150, 520],
        ),
        ConflictScenario(
            conflict_id="conflict-basic-03",
            topic="exercise timing",
            conflicting_statements=[
                "I usually hit the gym after work around 6pm.",
                "I switched to morning workouts. I exercise before work now.",
            ],
            correct_resolution=(
                "Alex switched from evening to morning workouts. Morning is current."
            ),
            resolution_type="explicit_correction",
            introduced_at_messages=[100, 700],
        ),
    ],
    # 5 temporal facts
    temporal_facts=[
        TemporalFact(
            fact_id="tf-basic-01",
            description="Alex lived in San Francisco",
            value="San Francisco, California",
            valid_from="2022-01-01",
            valid_until="2025-12-01",
            query_topic="residence history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-basic-02",
            description="Alex lives in Denver",
            value="Denver, Colorado",
            valid_from="2025-12-01",
            valid_until=None,
            query_topic="current city",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-basic-03",
            description="Alex was an omnivore",
            value="omnivore",
            valid_from=None,
            valid_until="2025-09-01",
            query_topic="diet history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-basic-04",
            description="Alex became vegetarian",
            value="vegetarian",
            valid_from="2025-09-01",
            valid_until=None,
            query_topic="current diet",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-basic-05",
            description="Alex is 28 years old",
            value="28",
            valid_from=None,
            valid_until=None,
            query_topic="age",
            should_be_current=True,
        ),
    ],
    # 10 query-relevance pairs
    query_relevance_pairs=[
        QueryRelevancePair(
            query_id="qrp-basic-01",
            query="What does Alex do for work?",
            expected_relevant_facts=["Data Analyst", "fintech startup"],
            expected_irrelevant_facts=["rock climbing", "Luna"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-02",
            query="Where does Alex live?",
            expected_relevant_facts=["Denver, Colorado", "moved from San Francisco"],
            expected_irrelevant_facts=["data analyst", "vegetarian"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-03",
            query="Does Alex have any pets?",
            expected_relevant_facts=["rescue cat named Luna", "considering second cat"],
            expected_irrelevant_facts=["Denver", "rock climbing"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-04",
            query="What are Alex's hobbies?",
            expected_relevant_facts=["rock climbing", "board games", "cooking Thai food"],
            expected_irrelevant_facts=["data analyst", "Denver"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-05",
            query="What is Alex's diet?",
            expected_relevant_facts=["vegetarian", "went vegetarian six months ago"],
            expected_irrelevant_facts=["rock climbing", "Luna"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-06",
            query="How old is Alex?",
            expected_relevant_facts=["28"],
            expected_irrelevant_facts=["Denver", "data analyst", "vegetarian"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-07",
            query="What is Alex's daily routine?",
            expected_relevant_facts=["exercises in the mornings", "before work"],
            expected_irrelevant_facts=["Luna", "Thai food"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-08",
            query="What are Alex's goals?",
            expected_relevant_facts=["learning Spanish", "Duolingo", "second cat"],
            expected_irrelevant_facts=["data analyst", "Denver"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-09",
            query="Is Alex in a relationship?",
            expected_relevant_facts=["single"],
            expected_irrelevant_facts=["Denver", "vegetarian", "rock climbing"],
        ),
        QueryRelevancePair(
            query_id="qrp-basic-10",
            query="How does Alex communicate?",
            expected_relevant_facts=["casual and friendly", "uses emojis"],
            expected_irrelevant_facts=["data analyst", "Luna"],
        ),
    ],
    forgettable_facts=[
        ForgettableFact(
            fact_id="ff-basic-01",
            text="Alex lived in San Francisco, California",
            reason="fully_superseded",
            mentioned_at_message=50,
            should_be_absent_after=210,
        ),
        ForgettableFact(
            fact_id="ff-basic-02",
            text="Alex was an omnivore",
            reason="fully_superseded",
            mentioned_at_message=30,
            should_be_absent_after=510,
        ),
        ForgettableFact(
            fact_id="ff-basic-03",
            text="Alex is feeling stressed about the move",
            reason="ephemeral_state",
            mentioned_at_message=190,
            should_be_absent_after=300,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Canonical Persona: Intermediate — Sarah Miller
# ---------------------------------------------------------------------------

PERSONA_INTERMEDIATE = RichPersonaSpec(
    persona_id="persona-2-intermediate",
    name="Sarah Miller",
    description=(
        "A 35-year-old marketing director who transitions to freelance consulting. "
        "She moves from Chicago to Portland, Oregon. She is married with a "
        "5-year-old daughter, trains for a marathon then switches to cycling, "
        "and starts a podcast about marketing trends. This persona tests "
        "richer relationships and more life transitions."
    ),
    complexity_level="intermediate",
    simulated_days=60,
    target_message_count=2000,
    # 14 profile dimensions covering WHO/WHAT/WHERE/WHEN/WHY/HOW
    profile_dimensions={
        # WHO
        "name": ProfileDimension(
            dimension_name="name",
            value="Sarah Miller",
            query_topic="name",
            category="who",
        ),
        "age": ProfileDimension(
            dimension_name="age",
            value="35",
            query_topic="age",
            category="who",
        ),
        "family_status": ProfileDimension(
            dimension_name="family_status",
            value="married with one daughter (age 5, named Emma)",
            query_topic="family",
            category="who",
        ),
        "nationality": ProfileDimension(
            dimension_name="nationality",
            value="American",
            query_topic="nationality",
            category="who",
        ),
        # WHAT
        "occupation": ProfileDimension(
            dimension_name="occupation",
            value="Freelance Marketing Consultant",
            query_topic="current occupation",
            category="what",
        ),
        "previous_occupation": ProfileDimension(
            dimension_name="previous_occupation",
            value="Marketing Director at a mid-size agency",
            query_topic="previous career",
            category="what",
        ),
        "hobbies": ProfileDimension(
            dimension_name="hobbies",
            value=["cycling", "writing", "photography"],
            query_topic="hobbies",
            category="what",
        ),
        "diet": ProfileDimension(
            dimension_name="diet",
            value="Mediterranean diet",
            query_topic="dietary preference",
            category="what",
        ),
        "pet": ProfileDimension(
            dimension_name="pet",
            value="a golden doodle named Biscuit",
            query_topic="pet",
            category="what",
        ),
        # WHERE
        "city": ProfileDimension(
            dimension_name="city",
            value="Portland, Oregon",
            query_topic="current city",
            category="where",
        ),
        "neighborhood": ProfileDimension(
            dimension_name="neighborhood",
            value="Alberta Arts District",
            query_topic="neighborhood",
            category="where",
        ),
        # WHEN
        "schedule": ProfileDimension(
            dimension_name="schedule",
            value="works mornings, picks up Emma from school at 3pm, cycles in the evening",
            query_topic="daily schedule",
            category="when",
        ),
        # WHY
        "motivations": ProfileDimension(
            dimension_name="motivations",
            value=(
                "writing a book about digital marketing, "
                "building her consulting brand, work-life balance"
            ),
            query_topic="motivations and goals",
            category="why",
        ),
        # HOW
        "communication_style": ProfileDimension(
            dimension_name="communication_style",
            value="professional but warm, uses structured lists, prefers concise answers",
            query_topic="communication style",
            category="how",
        ),
    },
    # 5 belief changes
    belief_changes=[
        BeliefChange(
            fact="occupation",
            old_value="Marketing Director at a mid-size agency",
            new_value="Freelance Marketing Consultant",
            query_topic="current occupation",
            changed_around_msg=500,
            key_messages=[470, 490, 510],
        ),
        BeliefChange(
            fact="city",
            old_value="Chicago, Illinois",
            new_value="Portland, Oregon",
            query_topic="current city",
            changed_around_msg=800,
            key_messages=[770, 790, 820],
        ),
        BeliefChange(
            fact="diet",
            old_value="keto diet",
            new_value="Mediterranean diet",
            query_topic="dietary preference",
            changed_around_msg=1200,
            key_messages=[1170, 1195, 1220],
        ),
        BeliefChange(
            fact="exercise",
            old_value="marathon training (running)",
            new_value="cycling",
            query_topic="exercise preference",
            changed_around_msg=1400,
            key_messages=[1380, 1400, 1420],
        ),
        BeliefChange(
            fact="side_project",
            old_value="writing a book about digital marketing",
            new_value="hosting a podcast about marketing trends and writing a book",
            query_topic="side projects",
            changed_around_msg=1700,
            key_messages=[1680, 1700, 1720],
        ),
    ],
    # 15 noise examples
    noise_examples=[
        NoiseExample(
            text="Good morning! Ready for another day.",
            reason="Generic greeting with no factual content",
        ),
        NoiseExample(
            text="Can you recommend a good movie to watch tonight?",
            reason="Entertainment request unrelated to persona attributes",
        ),
        NoiseExample(
            text="That makes sense, thanks for explaining!",
            reason="Acknowledgment with no persona information",
        ),
        NoiseExample(
            text="What time is it in Tokyo right now?",
            reason="Factual question unrelated to the persona",
        ),
        NoiseExample(
            text="Ugh, Mondays are the worst.",
            reason="Generic complaint with no factual content",
        ),
        NoiseExample(
            text="Can you summarize this article for me?",
            reason="Task request with no persona information",
        ),
        NoiseExample(
            text="Thanks, you're a lifesaver!",
            reason="Gratitude expression with no factual data",
        ),
        NoiseExample(
            text="What's the capital of Mongolia?",
            reason="Trivia question unrelated to persona",
        ),
        NoiseExample(
            text="I'm so tired today. Need more coffee.",
            reason="Casual complaint with minimal persona data",
        ),
        NoiseExample(
            text="That's a great idea, I'll try that!",
            reason="Positive reaction with no informational content",
        ),
        NoiseExample(
            text="Can you translate this to French?",
            reason="Translation request unrelated to persona",
        ),
        NoiseExample(
            text="Never mind, I figured it out.",
            reason="Dismissal with no persona information",
        ),
        NoiseExample(
            text="What's trending on Twitter today?",
            reason="Social media inquiry unrelated to persona",
        ),
        NoiseExample(
            text="Oops, wrong chat!",
            reason="Error message with no content",
        ),
        NoiseExample(
            text="BRB, need to grab something.",
            reason="Away notice with no factual content",
        ),
    ],
    # 15 signal examples
    signal_examples=[
        SignalExample(
            text=(
                "I've been a marketing director at this agency in Chicago for the past seven years."
            ),
            target_fact="occupation: Marketing Director, city: Chicago",
        ),
        SignalExample(
            text=(
                "Big news \u2014 I just put in my notice! "
                "I'm going freelance as a marketing consultant."
            ),
            target_fact="occupation change: Marketing Director → Freelance Consultant",
        ),
        SignalExample(
            text="We just closed on a house in Portland! The Alberta Arts District is so charming.",
            target_fact="city change: Chicago → Portland, neighborhood: Alberta Arts District",
        ),
        SignalExample(
            text=(
                "My husband and I are so excited about the move. "
                "Emma's a bit nervous about her new school though."
            ),
            target_fact="family: married, daughter named Emma",
        ),
        SignalExample(
            text="Biscuit, our golden doodle, has been such a good boy during the move.",
            target_fact="pet: golden doodle named Biscuit",
        ),
        SignalExample(
            text="I'm training for the Chicago marathon this October. Did 18 miles yesterday!",
            target_fact="exercise: marathon training, running",
        ),
        SignalExample(
            text=(
                "Had to stop running \u2014 my knee is a mess. "
                "Doctor said cycling would be better for me."
            ),
            target_fact="exercise change: running → cycling (knee injury)",
        ),
        SignalExample(
            text=(
                "I dropped keto. It was too restrictive. "
                "Switched to Mediterranean diet and I feel way better."
            ),
            target_fact="diet change: keto → Mediterranean",
        ),
        SignalExample(
            text=(
                "I just recorded the first episode of my podcast! "
                "It's about marketing trends in the creator economy."
            ),
            target_fact="side project: podcast about marketing trends",
        ),
        SignalExample(
            text=(
                "Still working on my book about digital marketing. "
                "The chapter on influencer ROI is killing me."
            ),
            target_fact="side project: writing a book about digital marketing",
        ),
        SignalExample(
            text=(
                "I'm 35 and finally feel like I'm building something "
                "that's mine with this consulting business."
            ),
            target_fact="age: 35, occupation: consulting",
        ),
        SignalExample(
            text=(
                "I work mornings while Emma's at school, pick her "
                "up at 3, then cycle in the evening."
            ),
            target_fact="schedule: work mornings, pick up Emma at 3pm, cycle evenings",
        ),
        SignalExample(
            text=(
                "I love photography, especially street photography. Portland has such great light."
            ),
            target_fact="hobbies: photography",
        ),
        SignalExample(
            text="Work-life balance is everything to me now. That's why I left the agency world.",
            target_fact="motivations: work-life balance",
        ),
        SignalExample(
            text=(
                "I prefer when you give me structured lists "
                "\u2014 bullet points over paragraphs please!"
            ),
            target_fact="communication style: prefers structured lists, concise",
        ),
    ],
    # 5 conflicts
    conflicts=[
        ConflictScenario(
            conflict_id="conflict-int-01",
            topic="exercise preference",
            conflicting_statements=[
                "I run every morning. Marathon training is my life right now.",
                "I've completely stopped running. Cycling is so much better for my knees.",
            ],
            correct_resolution=(
                "Sarah stopped running due to a knee injury and switched to cycling."
            ),
            resolution_type="explicit_correction",
            introduced_at_messages=[300, 1450],
        ),
        ConflictScenario(
            conflict_id="conflict-int-02",
            topic="location",
            conflicting_statements=[
                "Chicago is the best city. I could never leave.",
                "We just moved to Portland and I can't imagine ever going back.",
            ],
            correct_resolution="Sarah moved from Chicago to Portland. Portland is current.",
            resolution_type="recency",
            introduced_at_messages=[100, 850],
        ),
        ConflictScenario(
            conflict_id="conflict-int-03",
            topic="diet",
            conflicting_statements=[
                "Keto has changed my life. I've never felt better.",
                "I dropped keto completely. Mediterranean diet is the way to go.",
            ],
            correct_resolution=(
                "Sarah switched from keto to Mediterranean diet. Mediterranean is current."
            ),
            resolution_type="explicit_correction",
            introduced_at_messages=[200, 1250],
        ),
        ConflictScenario(
            conflict_id="conflict-int-04",
            topic="career satisfaction",
            conflicting_statements=[
                "I love my job at the agency. The team is amazing.",
                "Leaving the agency was the best decision I ever made. Freelance is freedom.",
            ],
            correct_resolution="Sarah left the agency to go freelance. She prefers freelance now.",
            resolution_type="recency",
            introduced_at_messages=[50, 600],
        ),
        ConflictScenario(
            conflict_id="conflict-int-05",
            topic="podcast",
            conflicting_statements=[
                "I don't think I'd ever start a podcast. Too much work.",
                "Just recorded episode 5 of my marketing podcast! The audience is growing.",
            ],
            correct_resolution=(
                "Sarah changed her mind and started a podcast about marketing trends."
            ),
            resolution_type="recency",
            introduced_at_messages=[400, 1750],
        ),
    ],
    # 8 temporal facts
    temporal_facts=[
        TemporalFact(
            fact_id="tf-int-01",
            description="Sarah worked as Marketing Director",
            value="Marketing Director at a mid-size agency",
            valid_from="2019-01-01",
            valid_until="2026-01-15",
            query_topic="occupation history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-int-02",
            description="Sarah works as Freelance Consultant",
            value="Freelance Marketing Consultant",
            valid_from="2026-01-15",
            valid_until=None,
            query_topic="current occupation",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-int-03",
            description="Sarah lived in Chicago",
            value="Chicago, Illinois",
            valid_from="2015-01-01",
            valid_until="2026-02-01",
            query_topic="residence history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-int-04",
            description="Sarah lives in Portland",
            value="Portland, Oregon",
            valid_from="2026-02-01",
            valid_until=None,
            query_topic="current city",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-int-05",
            description="Sarah followed keto diet",
            value="keto diet",
            valid_from="2024-06-01",
            valid_until="2026-02-15",
            query_topic="diet history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-int-06",
            description="Sarah follows Mediterranean diet",
            value="Mediterranean diet",
            valid_from="2026-02-15",
            valid_until=None,
            query_topic="current diet",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-int-07",
            description="Sarah trained for marathon",
            value="marathon training (running)",
            valid_from="2025-06-01",
            valid_until="2026-03-01",
            query_topic="exercise history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-int-08",
            description="Sarah switched to cycling",
            value="cycling",
            valid_from="2026-03-01",
            valid_until=None,
            query_topic="current exercise",
            should_be_current=True,
        ),
    ],
    # 15 query-relevance pairs
    query_relevance_pairs=[
        QueryRelevancePair(
            query_id="qrp-int-01",
            query="What does Sarah do for a living?",
            expected_relevant_facts=[
                "Freelance Marketing Consultant",
                "previously Marketing Director",
            ],
            expected_irrelevant_facts=["Mediterranean diet", "photography"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-02",
            query="Where does Sarah live?",
            expected_relevant_facts=[
                "Portland, Oregon",
                "Alberta Arts District",
                "previously Chicago",
            ],
            expected_irrelevant_facts=["marketing consultant", "cycling"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-03",
            query="Tell me about Sarah's family.",
            expected_relevant_facts=[
                "married",
                "daughter Emma, age 5",
            ],
            expected_irrelevant_facts=["freelance consultant", "Mediterranean diet"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-04",
            query="What is Sarah's diet?",
            expected_relevant_facts=[
                "Mediterranean diet",
                "previously keto",
            ],
            expected_irrelevant_facts=["Portland", "marketing consultant"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-05",
            query="Does Sarah exercise?",
            expected_relevant_facts=[
                "cycling",
                "previously marathon training",
                "knee injury",
            ],
            expected_irrelevant_facts=["Mediterranean diet", "podcast"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-06",
            query="Does Sarah have any pets?",
            expected_relevant_facts=["golden doodle named Biscuit"],
            expected_irrelevant_facts=["Portland", "cycling", "marketing"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-07",
            query="What are Sarah's side projects?",
            expected_relevant_facts=[
                "podcast about marketing trends",
                "writing a book about digital marketing",
            ],
            expected_irrelevant_facts=["Biscuit", "Portland"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-08",
            query="What are Sarah's hobbies?",
            expected_relevant_facts=["cycling", "writing", "photography"],
            expected_irrelevant_facts=["marketing consultant", "Emma"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-09",
            query="Why did Sarah leave her agency job?",
            expected_relevant_facts=[
                "work-life balance",
                "went freelance",
            ],
            expected_irrelevant_facts=["Mediterranean diet", "Biscuit"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-10",
            query="How does Sarah prefer to receive information?",
            expected_relevant_facts=[
                "structured lists",
                "bullet points",
                "concise answers",
            ],
            expected_irrelevant_facts=["Portland", "marathon"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-11",
            query="What is Sarah's daily schedule?",
            expected_relevant_facts=[
                "works mornings",
                "picks up Emma at 3pm",
                "cycles in the evening",
            ],
            expected_irrelevant_facts=["Mediterranean diet", "Biscuit"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-12",
            query="How old is Sarah?",
            expected_relevant_facts=["35"],
            expected_irrelevant_facts=["Portland", "marketing", "cycling"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-13",
            query="What neighborhood does Sarah live in?",
            expected_relevant_facts=["Alberta Arts District", "Portland"],
            expected_irrelevant_facts=["marketing consultant", "Mediterranean"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-14",
            query="Has Sarah's career changed recently?",
            expected_relevant_facts=[
                "left agency to go freelance",
                "Marketing Director to Freelance Consultant",
            ],
            expected_irrelevant_facts=["cycling", "Biscuit", "Mediterranean"],
        ),
        QueryRelevancePair(
            query_id="qrp-int-15",
            query="What is Sarah's nationality?",
            expected_relevant_facts=["American"],
            expected_irrelevant_facts=["Portland", "marketing", "keto"],
        ),
    ],
    forgettable_facts=[
        ForgettableFact(
            fact_id="ff-int-01",
            text="Sarah was a Marketing Director at an agency in Chicago",
            reason="fully_superseded",
            mentioned_at_message=100,
            should_be_absent_after=800,
        ),
        ForgettableFact(
            fact_id="ff-int-02",
            text="Sarah lived in Chicago",
            reason="fully_superseded",
            mentioned_at_message=50,
            should_be_absent_after=700,
        ),
        ForgettableFact(
            fact_id="ff-int-03",
            text="Sarah was training for a marathon",
            reason="fully_superseded",
            mentioned_at_message=300,
            should_be_absent_after=1200,
        ),
        ForgettableFact(
            fact_id="ff-int-04",
            text="Sarah followed a keto diet",
            reason="fully_superseded",
            mentioned_at_message=200,
            should_be_absent_after=1000,
        ),
        ForgettableFact(
            fact_id="ff-int-05",
            text="Sarah is nervous about leaving her agency job",
            reason="ephemeral_state",
            mentioned_at_message=650,
            should_be_absent_after=900,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Canonical Persona: Advanced — Marcus Rivera
# ---------------------------------------------------------------------------

PERSONA_ADVANCED = RichPersonaSpec(
    persona_id="persona-3-advanced",
    name="Marcus Rivera",
    description=(
        "A 41-year-old architect who goes through major life changes: divorce, "
        "career pivot to sustainable building, moves from NYC to Austin to "
        "rural Vermont, adopts and then modifies a vegan diet, takes up "
        "woodworking, beekeeping, and meditation, and publishes a blog about "
        "sustainable living. This persona tests complex multi-entity scenarios, "
        "frequent conflicts, and entity evolution."
    ),
    complexity_level="advanced",
    simulated_days=120,
    target_message_count=3000,
    # 18 profile dimensions covering WHO/WHAT/WHERE/WHEN/WHY/HOW
    profile_dimensions={
        # WHO (4)
        "name": ProfileDimension(
            dimension_name="name",
            value="Marcus Rivera",
            query_topic="name",
            category="who",
        ),
        "age": ProfileDimension(
            dimension_name="age",
            value="41",
            query_topic="age",
            category="who",
        ),
        "family_status": ProfileDimension(
            dimension_name="family_status",
            value="divorced, in a new relationship with Elena, two children (Mateo 14, Sofia 10)",
            query_topic="family and relationship status",
            category="who",
        ),
        "ethnicity": ProfileDimension(
            dimension_name="ethnicity",
            value="Mexican-American",
            query_topic="ethnicity or background",
            category="who",
        ),
        # WHAT (6)
        "occupation": ProfileDimension(
            dimension_name="occupation",
            value="Sustainable Building Consultant",
            query_topic="current occupation",
            category="what",
        ),
        "occupation_history": ProfileDimension(
            dimension_name="occupation_history",
            value=["Corporate Architect at a NYC firm", "Sustainable Building Consultant"],
            query_topic="career history",
            category="what",
        ),
        "hobbies": ProfileDimension(
            dimension_name="hobbies",
            value=["woodworking", "beekeeping", "meditation", "hiking"],
            query_topic="hobbies",
            category="what",
        ),
        "diet": ProfileDimension(
            dimension_name="diet",
            value="flexitarian (mostly plant-based, occasional fish and eggs)",
            query_topic="diet",
            category="what",
        ),
        "health": ProfileDimension(
            dimension_name="health",
            value="practices daily meditation, recovered from burnout",
            query_topic="health and wellness",
            category="what",
        ),
        "side_projects": ProfileDimension(
            dimension_name="side_projects",
            value=["blog about sustainable living", "building an off-grid cabin"],
            query_topic="side projects",
            category="what",
        ),
        # WHERE (3)
        "city": ProfileDimension(
            dimension_name="city",
            value="rural Vermont (near Woodstock)",
            query_topic="current location",
            category="where",
        ),
        "property": ProfileDimension(
            dimension_name="property",
            value="15-acre rural property with a farmhouse and workshop",
            query_topic="property",
            category="where",
        ),
        "commute": ProfileDimension(
            dimension_name="commute",
            value="works from home, travels to project sites occasionally",
            query_topic="commute",
            category="where",
        ),
        # WHEN (1)
        "schedule": ProfileDimension(
            dimension_name="schedule",
            value=(
                "meditates at sunrise, works 9-3, "
                "woodworking/beekeeping afternoons, kids on weekends"
            ),
            query_topic="daily schedule",
            category="when",
        ),
        # WHY (2)
        "motivations": ProfileDimension(
            dimension_name="motivations",
            value="environmental sustainability, self-sufficiency, being present for his kids",
            query_topic="motivations and values",
            category="why",
        ),
        "political_views": ProfileDimension(
            dimension_name="political_views",
            value="environmental activist, previously moderate centrist",
            query_topic="political views",
            category="why",
        ),
        # HOW (2)
        "communication_style": ProfileDimension(
            dimension_name="communication_style",
            value="thoughtful and deliberate, avoids small talk, prefers deep conversations",
            query_topic="communication style",
            category="how",
        ),
        "tech_preferences": ProfileDimension(
            dimension_name="tech_preferences",
            value=(
                "minimal tech use, avoids social media except for blog, "
                "prefers phone calls over texts"
            ),
            query_topic="technology preferences",
            category="how",
        ),
    },
    # 7 belief changes
    belief_changes=[
        BeliefChange(
            fact="occupation",
            old_value="Corporate Architect at a NYC firm",
            new_value="Sustainable Building Consultant",
            query_topic="current occupation",
            changed_around_msg=600,
            key_messages=[570, 590, 620],
        ),
        BeliefChange(
            fact="city",
            old_value="New York City",
            new_value="Austin, Texas",
            query_topic="current location",
            changed_around_msg=900,
            key_messages=[870, 890, 920],
        ),
        BeliefChange(
            fact="city",
            old_value="Austin, Texas",
            new_value="rural Vermont (near Woodstock)",
            query_topic="current location",
            changed_around_msg=1800,
            key_messages=[1770, 1800, 1830],
        ),
        BeliefChange(
            fact="relationship_status",
            old_value="married to Diana",
            new_value="divorced",
            query_topic="relationship status",
            changed_around_msg=1200,
            key_messages=[1170, 1200, 1230],
        ),
        BeliefChange(
            fact="relationship_status",
            old_value="divorced, single",
            new_value="in a new relationship with Elena",
            query_topic="relationship status",
            changed_around_msg=2400,
            key_messages=[2370, 2400, 2430],
        ),
        BeliefChange(
            fact="diet",
            old_value="standard American diet",
            new_value="strict vegan",
            query_topic="diet",
            changed_around_msg=1500,
            key_messages=[1470, 1500, 1530],
        ),
        BeliefChange(
            fact="diet",
            old_value="strict vegan",
            new_value="flexitarian (mostly plant-based, occasional fish and eggs)",
            query_topic="diet",
            changed_around_msg=2700,
            key_messages=[2670, 2700, 2730],
        ),
    ],
    # 20 noise examples
    noise_examples=[
        NoiseExample(
            text="Hey, how's it going?",
            reason="Generic greeting with no factual content",
        ),
        NoiseExample(
            text="What do you think about the new iPhone?",
            reason="Product inquiry unrelated to persona",
        ),
        NoiseExample(
            text="Can you help me draft an email?",
            reason="Task request with no persona data",
        ),
        NoiseExample(
            text="I keep forgetting my passwords. Any tips?",
            reason="Generic help request",
        ),
        NoiseExample(
            text="Ha, that's a good one! You're getting funnier.",
            reason="Social reaction with no informational content",
        ),
        NoiseExample(
            text="What time does the hardware store close?",
            reason="Factual question unrelated to persona traits",
        ),
        NoiseExample(
            text="Can you convert 50 euros to dollars?",
            reason="Currency conversion unrelated to persona",
        ),
        NoiseExample(
            text="Thanks, that was helpful.",
            reason="Gratitude expression with no factual data",
        ),
        NoiseExample(
            text="What's the best way to clean a cast iron skillet?",
            reason="Cooking question unrelated to persona identity",
        ),
        NoiseExample(
            text="Never mind, I'll figure it out.",
            reason="Dismissal with no persona information",
        ),
        NoiseExample(
            text="Hmm, interesting. Let me think about it.",
            reason="Filler with no factual content",
        ),
        NoiseExample(
            text="What's the capital of Peru?",
            reason="Trivia unrelated to persona",
        ),
        NoiseExample(
            text="Can you set a timer for 20 minutes?",
            reason="Utility request with no persona data",
        ),
        NoiseExample(
            text="That's wild. I had no idea.",
            reason="Reaction with no informational content",
        ),
        NoiseExample(
            text="What's the difference between affect and effect?",
            reason="Grammar question unrelated to persona",
        ),
        NoiseExample(
            text="I'll deal with that later.",
            reason="Procrastination statement with no factual content",
        ),
        NoiseExample(
            text="Can you recommend a good podcast?",
            reason="Recommendation request unrelated to persona traits",
        ),
        NoiseExample(
            text="LOL okay fair enough.",
            reason="Casual response with no persona data",
        ),
        NoiseExample(
            text="What's the weather forecast for next week?",
            reason="Weather inquiry unrelated to persona",
        ),
        NoiseExample(
            text="Sorry, I got sidetracked. What were we talking about?",
            reason="Conversational filler with no factual content",
        ),
    ],
    # 20 signal examples
    signal_examples=[
        SignalExample(
            text=(
                "I've been an architect at this firm in Manhattan "
                "for 12 years. Corporate towers mostly."
            ),
            target_fact="occupation: Corporate Architect in NYC",
        ),
        SignalExample(
            text=(
                "I just resigned. I'm pivoting to sustainable "
                "building consulting. It's what I believe in."
            ),
            target_fact="occupation change: Corporate Architect → Sustainable Building Consultant",
        ),
        SignalExample(
            text="Diana and I are getting divorced. It's been coming for a while.",
            target_fact="relationship change: married → divorcing",
        ),
        SignalExample(
            text="I met someone. Her name is Elena. She's a botanist and we just... click.",
            target_fact="new relationship with Elena",
        ),
        SignalExample(
            text="We moved to Austin. Needed a fresh start after the divorce.",
            target_fact="city change: NYC → Austin",
        ),
        SignalExample(
            text=(
                "We bought 15 acres in Vermont near Woodstock. "
                "I'm building our life here from scratch."
            ),
            target_fact="city change: Austin → rural Vermont, property: 15 acres",
        ),
        SignalExample(
            text=(
                "Mateo is 14 now, Sofia just turned 10. "
                "They come up on weekends. It's the best part of my week."
            ),
            target_fact="children: Mateo 14, Sofia 10, custody weekends",
        ),
        SignalExample(
            text=(
                "I went fully vegan six months ago. For the planet, for my health, for the animals."
            ),
            target_fact="diet change: standard → vegan",
        ),
        SignalExample(
            text=(
                "I've relaxed the vegan thing a bit. I eat eggs from "
                "our chickens and fish occasionally. Flexitarian I guess."
            ),
            target_fact="diet change: vegan → flexitarian",
        ),
        SignalExample(
            text=(
                "I built a proper woodworking shop in the barn. "
                "Made my first dining table last month."
            ),
            target_fact="hobby: woodworking, property: workshop in barn",
        ),
        SignalExample(
            text=(
                "I started beekeeping this spring. We have three "
                "hives now and the honey is incredible."
            ),
            target_fact="hobby: beekeeping",
        ),
        SignalExample(
            text="I meditate every morning at sunrise. It saved me during the divorce honestly.",
            target_fact="hobby/health: meditation at sunrise",
        ),
        SignalExample(
            text=(
                "I'm building an off-grid cabin on the back of the "
                "property. Solar panels, composting toilet, the works."
            ),
            target_fact="side project: off-grid cabin",
        ),
        SignalExample(
            text="I started a blog about sustainable living. It's getting some traction actually.",
            target_fact="side project: sustainability blog",
        ),
        SignalExample(
            text=(
                "My politics have shifted a lot. I used to be a moderate "
                "centrist. Now I'm a full-on environmental activist."
            ),
            target_fact="political change: moderate → environmental activist",
        ),
        SignalExample(
            text=(
                "I barely use social media anymore. Just the blog. "
                "I prefer phone calls over texts honestly."
            ),
            target_fact="tech preferences: minimal social media, prefers phone calls",
        ),
        SignalExample(
            text=(
                "I'm 41 and I feel like I'm finally living authentically. "
                "The corporate world almost broke me."
            ),
            target_fact="age: 41, burnout recovery",
        ),
        SignalExample(
            text=(
                "My dad was Mexican, mom was American. Growing up "
                "bicultural shaped everything about how I see the world."
            ),
            target_fact="ethnicity: Mexican-American",
        ),
        SignalExample(
            text="I work from home now, 9 to 3, then I'm in the workshop or with the bees.",
            target_fact="schedule: work from home 9-3, afternoons woodworking/beekeeping",
        ),
        SignalExample(
            text=(
                "Sustainability isn't just my job, it's my whole life "
                "philosophy now. Self-sufficiency, being present for the kids."
            ),
            target_fact="motivations: sustainability, self-sufficiency, presence for kids",
        ),
    ],
    # 8 conflicts
    conflicts=[
        ConflictScenario(
            conflict_id="conflict-adv-01",
            topic="diet",
            conflicting_statements=[
                "I'm fully vegan. No animal products whatsoever. It's a moral commitment.",
                "I had eggs from our chickens this morning and "
                "grilled fish last night. I'm more flexitarian now.",
            ],
            correct_resolution=(
                "Marcus was strict vegan but relaxed to flexitarian. "
                "He now eats eggs from his own chickens and occasional fish."
            ),
            resolution_type="recency",
            introduced_at_messages=[1520, 2720],
        ),
        ConflictScenario(
            conflict_id="conflict-adv-02",
            topic="social media stance",
            conflicting_statements=[
                "Social media is toxic. I deleted all my accounts.",
                "My blog has an Instagram presence now. You have to be where people are.",
            ],
            correct_resolution=(
                "Marcus initially rejected social media but now uses it "
                "selectively for his sustainability blog."
            ),
            resolution_type="explicit_correction",
            introduced_at_messages=[1000, 2500],
        ),
        ConflictScenario(
            conflict_id="conflict-adv-03",
            topic="relationship status",
            conflicting_statements=[
                "I'm done with relationships. Being single is the "
                "healthiest thing for me right now.",
                "Elena and I just celebrated six months together. She moved in last week.",
            ],
            correct_resolution=(
                "Marcus was single after his divorce but started a new relationship with Elena."
            ),
            resolution_type="recency",
            introduced_at_messages=[1300, 2450],
        ),
        ConflictScenario(
            conflict_id="conflict-adv-04",
            topic="city preference",
            conflicting_statements=[
                "I love Austin. The energy here is exactly what I needed.",
                "Austin got too big and commercial. Vermont is where I belong.",
            ],
            correct_resolution=(
                "Marcus moved from Austin to Vermont. He grew disillusioned "
                "with Austin and prefers rural Vermont now."
            ),
            resolution_type="recency",
            introduced_at_messages=[950, 1850],
        ),
        ConflictScenario(
            conflict_id="conflict-adv-05",
            topic="career satisfaction",
            conflicting_statements=[
                "Architecture is my calling. I can't imagine doing anything else.",
                "I left corporate architecture and I've never been "
                "happier. Consulting on sustainable projects is the "
                "real work.",
            ],
            correct_resolution=(
                "Marcus transitioned from corporate architecture to sustainable "
                "building consulting and is happier."
            ),
            resolution_type="recency",
            introduced_at_messages=[200, 650],
        ),
        ConflictScenario(
            conflict_id="conflict-adv-06",
            topic="remote work opinion",
            conflicting_statements=[
                "Remote work doesn't work for architecture. "
                "You need to be in the office with the team.",
                "Working from home on my Vermont property is the most productive I've ever been.",
            ],
            correct_resolution=(
                "Marcus changed his view on remote work after leaving corporate. "
                "He now works from home and finds it productive."
            ),
            resolution_type="explicit_correction",
            introduced_at_messages=[300, 2000],
        ),
        ConflictScenario(
            conflict_id="conflict-adv-07",
            topic="political engagement",
            conflicting_statements=[
                "I stay out of politics. Both sides are equally frustrating.",
                "I spoke at the town hall about the new solar "
                "initiative. Environmental policy is too important "
                "to sit out.",
            ],
            correct_resolution=(
                "Marcus went from politically disengaged moderate to active environmental advocate."
            ),
            resolution_type="recency",
            introduced_at_messages=[400, 2200],
        ),
        ConflictScenario(
            conflict_id="conflict-adv-08",
            topic="technology usage",
            conflicting_statements=[
                "I love my smart home setup. Every room has Alexa and automated lights.",
                "I ripped out all the smart home stuff. I want to "
                "live simply. Less tech, more presence.",
            ],
            correct_resolution=(
                "Marcus moved from tech-heavy to minimal tech lifestyle as part "
                "of his sustainability and simplicity philosophy."
            ),
            resolution_type="explicit_correction",
            introduced_at_messages=[150, 1900],
        ),
    ],
    # 12 temporal facts
    temporal_facts=[
        TemporalFact(
            fact_id="tf-adv-01",
            description="Marcus worked as corporate architect in NYC",
            value="Corporate Architect at a NYC firm",
            valid_from="2013-01-01",
            valid_until="2025-06-01",
            query_topic="occupation history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-adv-02",
            description="Marcus works as sustainable building consultant",
            value="Sustainable Building Consultant",
            valid_from="2025-06-01",
            valid_until=None,
            query_topic="current occupation",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-adv-03",
            description="Marcus lived in New York City",
            value="New York City",
            valid_from="2010-01-01",
            valid_until="2025-08-01",
            query_topic="residence history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-adv-04",
            description="Marcus lived in Austin",
            value="Austin, Texas",
            valid_from="2025-08-01",
            valid_until="2026-01-01",
            query_topic="residence history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-adv-05",
            description="Marcus lives in rural Vermont",
            value="rural Vermont (near Woodstock)",
            valid_from="2026-01-01",
            valid_until=None,
            query_topic="current location",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-adv-06",
            description="Marcus was married to Diana",
            value="married to Diana",
            valid_from="2012-01-01",
            valid_until="2025-10-01",
            query_topic="relationship history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-adv-07",
            description="Marcus was divorced and single",
            value="divorced, single",
            valid_from="2025-10-01",
            valid_until="2026-02-01",
            query_topic="relationship history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-adv-08",
            description="Marcus is in a relationship with Elena",
            value="in a new relationship with Elena",
            valid_from="2026-02-01",
            valid_until=None,
            query_topic="current relationship status",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-adv-09",
            description="Marcus followed standard American diet",
            value="standard American diet",
            valid_from=None,
            valid_until="2025-09-01",
            query_topic="diet history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-adv-10",
            description="Marcus was strict vegan",
            value="strict vegan",
            valid_from="2025-09-01",
            valid_until="2026-02-15",
            query_topic="diet history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-adv-11",
            description="Marcus follows flexitarian diet",
            value="flexitarian (mostly plant-based, occasional fish and eggs)",
            valid_from="2026-02-15",
            valid_until=None,
            query_topic="current diet",
            should_be_current=True,
        ),
        TemporalFact(
            fact_id="tf-adv-12",
            description="Marcus was politically moderate",
            value="moderate centrist",
            valid_from=None,
            valid_until="2025-12-01",
            query_topic="political history",
            should_be_current=False,
        ),
    ],
    # 20 query-relevance pairs
    query_relevance_pairs=[
        QueryRelevancePair(
            query_id="qrp-adv-01",
            query="What does Marcus do for work?",
            expected_relevant_facts=[
                "Sustainable Building Consultant",
                "previously corporate architect in NYC",
            ],
            expected_irrelevant_facts=["beekeeping", "flexitarian"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-02",
            query="Where does Marcus live?",
            expected_relevant_facts=[
                "rural Vermont near Woodstock",
                "15-acre property",
                "previously NYC and Austin",
            ],
            expected_irrelevant_facts=["sustainable building", "vegan"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-03",
            query="What is Marcus's relationship status?",
            expected_relevant_facts=[
                "in a relationship with Elena",
                "divorced from Diana",
            ],
            expected_irrelevant_facts=["woodworking", "Vermont"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-04",
            query="Does Marcus have children?",
            expected_relevant_facts=[
                "son Mateo, age 14",
                "daughter Sofia, age 10",
                "sees them on weekends",
            ],
            expected_irrelevant_facts=["beekeeping", "blog"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-05",
            query="What is Marcus's diet?",
            expected_relevant_facts=[
                "flexitarian",
                "mostly plant-based",
                "previously strict vegan",
            ],
            expected_irrelevant_facts=["architect", "Vermont"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-06",
            query="What are Marcus's hobbies?",
            expected_relevant_facts=[
                "woodworking",
                "beekeeping",
                "meditation",
                "hiking",
            ],
            expected_irrelevant_facts=["sustainable building", "Elena"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-07",
            query="What are Marcus's side projects?",
            expected_relevant_facts=[
                "blog about sustainable living",
                "building off-grid cabin",
            ],
            expected_irrelevant_facts=["Elena", "meditation"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-08",
            query="What are Marcus's political views?",
            expected_relevant_facts=[
                "environmental activist",
                "previously moderate centrist",
            ],
            expected_irrelevant_facts=["woodworking", "flexitarian"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-09",
            query="How does Marcus feel about technology?",
            expected_relevant_facts=[
                "minimal tech use",
                "avoids social media",
                "prefers phone calls",
            ],
            expected_irrelevant_facts=["beekeeping", "Elena"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-10",
            query="What is Marcus's daily routine?",
            expected_relevant_facts=[
                "meditates at sunrise",
                "works 9-3",
                "woodworking or beekeeping in afternoons",
            ],
            expected_irrelevant_facts=["Elena", "divorce"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-11",
            query="Where has Marcus lived?",
            expected_relevant_facts=[
                "New York City",
                "Austin, Texas",
                "rural Vermont",
            ],
            expected_irrelevant_facts=["meditation", "flexitarian"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-12",
            query="Why did Marcus leave corporate architecture?",
            expected_relevant_facts=[
                "burnout",
                "sustainability values",
                "wanted to live authentically",
            ],
            expected_irrelevant_facts=["Elena", "beekeeping"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-13",
            query="How does Marcus communicate?",
            expected_relevant_facts=[
                "thoughtful and deliberate",
                "avoids small talk",
                "prefers deep conversations",
            ],
            expected_irrelevant_facts=["architect", "Vermont"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-14",
            query="What is Marcus's ethnic background?",
            expected_relevant_facts=["Mexican-American", "bicultural"],
            expected_irrelevant_facts=["sustainable building", "vegan"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-15",
            query="Tell me about Marcus's property.",
            expected_relevant_facts=[
                "15-acre rural property",
                "farmhouse",
                "workshop in barn",
                "off-grid cabin",
            ],
            expected_irrelevant_facts=["divorce", "meditation"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-16",
            query="Has Marcus's diet changed?",
            expected_relevant_facts=[
                "standard American diet to vegan to flexitarian",
                "eats eggs from his own chickens",
                "occasional fish",
            ],
            expected_irrelevant_facts=["woodworking", "NYC"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-17",
            query="What happened with Marcus's marriage?",
            expected_relevant_facts=[
                "divorced from Diana",
                "now with Elena",
            ],
            expected_irrelevant_facts=["beekeeping", "blog"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-18",
            query="What motivates Marcus?",
            expected_relevant_facts=[
                "environmental sustainability",
                "self-sufficiency",
                "being present for his kids",
            ],
            expected_irrelevant_facts=["architect", "Austin"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-19",
            query="How old is Marcus?",
            expected_relevant_facts=["41"],
            expected_irrelevant_facts=["Vermont", "sustainable building", "Elena"],
        ),
        QueryRelevancePair(
            query_id="qrp-adv-20",
            query="Does Marcus use social media?",
            expected_relevant_facts=[
                "avoids social media",
                "blog has some social presence",
                "prefers phone calls",
            ],
            expected_irrelevant_facts=["woodworking", "flexitarian"],
        ),
    ],
    forgettable_facts=[
        ForgettableFact(
            fact_id="ff-adv-01",
            text="Marcus was married to Elena",
            reason="fully_superseded",
            mentioned_at_message=100,
            should_be_absent_after=600,
        ),
        ForgettableFact(
            fact_id="ff-adv-02",
            text="Marcus worked as an architect at a large firm in NYC",
            reason="fully_superseded",
            mentioned_at_message=50,
            should_be_absent_after=800,
        ),
        ForgettableFact(
            fact_id="ff-adv-03",
            text="Marcus lived in New York City",
            reason="fully_superseded",
            mentioned_at_message=30,
            should_be_absent_after=1000,
        ),
        ForgettableFact(
            fact_id="ff-adv-04",
            text="Marcus lived in Austin, Texas",
            reason="fully_superseded",
            mentioned_at_message=800,
            should_be_absent_after=2000,
        ),
        ForgettableFact(
            fact_id="ff-adv-05",
            text="Marcus followed a strict vegan diet",
            reason="fully_superseded",
            mentioned_at_message=600,
            should_be_absent_after=2200,
        ),
        ForgettableFact(
            fact_id="ff-adv-06",
            text="Marcus is feeling overwhelmed by the divorce",
            reason="ephemeral_state",
            mentioned_at_message=500,
            should_be_absent_after=800,
        ),
        ForgettableFact(
            fact_id="ff-adv-07",
            text="Marcus is temporarily staying at a friend's place in Austin",
            reason="session_context",
            mentioned_at_message=850,
            should_be_absent_after=1100,
        ),
    ],
)


# ---------------------------------------------------------------------------
# All canonical personas
# ---------------------------------------------------------------------------

ALL_PERSONAS: list[RichPersonaSpec] = [
    PERSONA_BASIC,
    PERSONA_INTERMEDIATE,
    PERSONA_ADVANCED,
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_persona_basic() -> RichPersonaSpec:
    """Return the basic-complexity persona specification (Alex Chen)."""
    return PERSONA_BASIC


def get_persona_intermediate() -> RichPersonaSpec:
    """Return the intermediate-complexity persona specification (Sarah Miller)."""
    return PERSONA_INTERMEDIATE


def get_persona_advanced() -> RichPersonaSpec:
    """Return the advanced-complexity persona specification (Marcus Rivera)."""
    return PERSONA_ADVANCED


__all__ = [
    "RichPersonaSpec",
    "PERSONA_BASIC",
    "PERSONA_INTERMEDIATE",
    "PERSONA_ADVANCED",
    "ALL_PERSONAS",
    "get_persona_basic",
    "get_persona_intermediate",
    "get_persona_advanced",
]
