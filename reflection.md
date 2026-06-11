# Reflection: Profile Comparisons

## High-Energy Pop vs. Chill Lofi

These two profiles are near-opposites, and the outputs reflect that cleanly. The pop profile surfaces fast, bright tracks (Sunrise City, Gym Hero, Festival Pulse) in the 120–130 BPM range with low acousticness. The lofi profile returns slow, mellow, acoustic-heavy tracks (Library Rain, Midnight Coding, Study Fog) around 70–80 BPM.
This makes sense: the genre and energy targets pull in exactly opposite directions, so almost no song can score well on both.

## Chill Lofi vs. Deep Intense Rock

Lofi tops out with a near-perfect 4.50 (Library Rain hits genre + mood + energy + acoustic). Rock tops out at 3.99. It's high, but the rock *runner-up* drops all the way to 1.98 because there's only one rock song in the catalog. After Storm Runner, the rock profile falls back to mood-only matches (Trap House Sunrise, Gym Hero, Basement Riot). This is the dataset-skew limitation showing up in practice.

## High-Energy Pop vs. Adversarial "Conflicted Listener"

Both profiles ask for pop and high energy, but the adversarial one swaps "happy" for "sad" (a mood that doesn't exist in the dataset). The top two results are nearly the same, Gym Hero and Sunrise City, because the scorer quietly ignores the unknown mood and lets genre + energy take over. This confirms the limitation in the model
card: strict equality on mood means "sad" contributes zero rather than reshaping the ranking.

## Weight-shift experiment (energy ×2, genre ÷2)

I temporarily flipped the weights so energy was worth +2.0 and genre only +1.0, then
re-ran all four profiles (weights have since been reverted).

- **Top-1 didn't move** for any profile — Sunrise City, Library Rain, Storm Runner,
  and Gym Hero all stayed #1. When a song already matches genre *and* mood *and*
  energy, no reasonable weight change can dislodge it.
- **The middle of the list opened up.** Cross-genre songs started breaking in: the
  Deep Intense Rock list now includes Trap House Sunrise above Gym Hero by energy
  proximity, and the adversarial pop list pulled in Storm Runner and Trap House
  Sunrise at ranks 3–4 — songs the original weights kept out entirely.
- **Takeaway:** genre weighting mainly controls *diversity* in the tail, not the
  top pick. Lowering it is a cheap way to let the recommender surface songs that
  feel right *acoustically* even when the label doesn't match.

## Plain-language takeaway

The reason "Gym Hero" keeps showing up for people who say they want "Happy Pop" is that the system rewards *genre* twice as hard as anything else. "Gym Hero" is pop and it's very energetic, so it racks up a big genre bonus plus a big energy bonus, enough to beat songs that are actually happy but belong to a different genre. In other words, the recommender has decided that being the right *kind* of music matters
more than being the right *mood*, and until we rebalance the weights (or let genres partially match), that's the behavior we'll keep getting.

---

# Reflection and Ethics

## Limitations and biases

The original scorer's limitations are documented in [model_card.md](model_card.md) §6 — genre dominance, dataset skew toward lofi, strict equality on categorical fields, and silent failure on out-of-vocabulary moods. The Phase 2 RAG layer added three new ones worth naming:

- **Descriptor authorship.** Each song's `descriptor` column is one prose line I wrote by hand. That means my vocabulary, my associations, and my blind spots are baked into what the embedding model can match. A song I described as "late-night driving" will surface for "night drive" queries; a song I described only as "groovy" won't, even if it's musically identical.
- **Embedding model bias.** MiniLM was trained on web text. It has whatever biases that corpus carries — including stronger associations for genres and moods that are overrepresented online (English-language pop, lofi study culture) and weaker ones for genres that aren't (regional, non-English, or niche scenes).
- **Parser scope.** The keyword/regex parser only knows the words I put in its dictionary. It is English-only and matches on surface forms, so "melancholy" parses to nothing while "moody" works. This silently routes ambiguous queries to retrieval-only ranking without telling the user.

## Misuse and prevention

Misuse surface for the local version is small *because* the system is small: no user accounts, no logged queries, no telemetry, no network calls past the one-time model download. There is no behavioral data to leak and no audience to target.

The scaled-up version of this architecture has one obvious misuse mode: **descriptors are author-controlled prose that drives retrieval**. In a multi-tenant deployment — say, artists or labels submitting their own songs and their own descriptor text — that's an SEO-style attack waiting to happen. Someone could write a descriptor stuffed with high-traffic query terms ("perfect for studying, gym, late-night driving") to game placement, and because retrieval is the *gate* to the ranker, a manipulated descriptor can outrank a more honest one.

Two prevention moves I'd want before scaling: (1) separate author-supplied marketing copy from a system-controlled "ground-truth" descriptor generated from audio features or moderated metadata, so the embedding sees the latter; (2) audit how much of a song's final score came from cosine vs. rule-based match, and surface anomalies (high cosine, low rule-match) for review.

## What surprised me in reliability testing

The Phase 4 held-out eval (12 queries, one per genre — see [eval/results.md](eval/results.md)) had two surprises.

The first was the "rainy night" query pulling *Velvet Hours*, *Coffee Shop Stories*, and *Library Rain* to the top despite none of those songs having "rainy" in their metadata. The embedding was reading mood from descriptor prose in a way no keyword match could replicate. I had expected retrieval to mostly act as a fancy fuzzy filter; it turned out to be doing genuine semantic work.

The second was that the parser fill rate of 10/12 didn't hurt the final hit rate at all — the two queries where the parser failed (`festival edm dance party`, `smooth r&b late night`) still landed correct genres in the top-5 because retrieval carried them. I had assumed parser failure would be a clean failure mode; instead the two layers covered for each other in a way I hadn't designed for.

## Collaboration with AI

I used Claude throughout the project for design conversation, scaffolding, and code review.

**Helpful suggestion:** Early in Phase 2, when I was scoping the RAG layer, Claude pushed back on my initial plan to *replace* the rule-based scorer with embedding similarity. The suggestion was to keep the scorer untouched and use retrieval only to choose which 15 songs the scorer ranks. That framing — "RAG layered on top of, not replacing, the deterministic logic" — became the load-bearing design decision for the whole project. It preserved the explainability the original assignment was built around while still making the AI feature genuinely necessary (without retrieval, no candidates flow into the ranker).

**Flawed suggestion:** During the four-profile evaluation, Claude was happy to generate plausible-sounding paragraphs explaining why specific songs ranked where they did, but more than once those paragraphs were wrong about the actual numeric breakdown. The model would confidently claim a song won on "mood + energy" when the score breakdown showed it was actually genre + energy with no mood match. This helped to reinforce in me that you should always double check and never paste an AI-generated explanation into the writeup without first verifying its accuracy. AI is reliable for *generating* explanations of code, but not for *interpreting* numeric output unless you are willing to recompute by hand. 
