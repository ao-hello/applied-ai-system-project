# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0** — a tiny content-based music recommender

---

## 2. Intended Use

VibeFinder suggests 5 songs from a small 20-track catalog based on a user's
preferred genre, mood, target energy level, and whether they like acoustic
music. It assumes the user can articulate their taste as a few discrete
preferences, and that the catalog actually contains something close to what
they want.

**Intended for:** classroom exploration of how content-based recommenders
turn features into rankings, and how small scoring choices produce large
behavioral changes.

**Not intended for:** real product use, large catalogs, users who can't (or
shouldn't have to) describe their taste in rigid categorical terms, or any
setting where fairness across genres/artists matters.

---

## 3. How the Model Works

Imagine each song has a label for its *genre* (pop, lofi, rock...) and its *mood* (happy, chill, intense...), plus a dial from 0 to 1 for how energetic it is and another dial for how acoustic it is. The user fills out a tiny form: favorite genre, favorite mood, a target energy level, and whether they like acoustic songs.

For every song in the catalog, VibeFinder starts at zero and adds points:

- **+2 points** if the song's genre matches the user's favorite.
- **+1 point** if the song's mood matches.
- **Up to +1 point** for being close to the user's target energy. A perfect match gets the full point, and the farther off the song is, the smaller the bonus.
- **+0.5 points** if the user likes acoustic music and this song is very
acoustic.

Every song gets a number, VibeFinder sorts them from highest to lowest, and the top 5 come back. That's the whole idea: score, sort, slice.

**Changes from the starter logic.** The starter only printed one list for
one profile. I extended `main.py` to loop over four profiles (three normal, one adversarial), print each result with a human-readable explanation, and kept the scoring weights tunable at the top of `recommender.py` so I could run a weight-shift experiment.

---

## 4. Data

`data/songs.csv` contains **20 songs**, unchanged from the starter. Each
row has: id, title, artist, genre, mood, and numeric features (energy,
tempo_bpm, valence, danceability, acousticness).

**Genres represented:** pop (2), lofi (4), rock (1), ambient (2), jazz (2),
synthwave (2), indie pop (2), punk (1), hip hop (1), edm (1), r&b (1),
folk (1). Lofi is overrepresented; most other genres have one or two tracks.

**Moods represented:** chill, happy, intense, moody, relaxed, focused. Notably
absent: sad, angry, romantic, nostalgic, melancholic — common moods listeners
actually use.

**What's missing:** No classical, country, metal, k-pop, latin, reggae, or
world music. No language or lyric signals. No tempo-based matching. No signal
for song length, era, or artist popularity. A real music-taste landscape is
far wider than this catalog can represent.

---

## 5. Strengths

- **Well-represented profiles work well:** The Chill Lofi profile produced a
  coherent, intuitive top 5 (Library Rain, Midnight Coding, Study Fog, Focus
  Flow, Spacewalk Thoughts). Lofi is the densest genre in the dataset, so the
  recommender has enough material to actually *rank* within taste, not just
  retrieve.
- **Transparent explanations:** Every recommendation comes with a
  human-readable "Because: genre match (pop) +2.0; mood match (happy) +1.0;
  energy sim +0.97" line. A user can always see why a song ranked where it
  did — a property that expensive deep-learning recommenders often lack.
- **Energy similarity is well-behaved:** Rewarding proximity (rather than
  "higher is better") captures the fact that a chill listener wants
  low-energy songs and a gym listener wants high-energy ones. The same
  formula handles both without special cases.
- **Stable top-1:** Across the weight-shift experiment, the #1 pick never
  moved for any profile, which suggests the model genuinely agrees on "best
  fit" when a song matches on all three signals.

---

## 6. Limitations and Bias

**Genre dominance:** Because the genre match is worth +2.0 (double any other signal),
a single genre hit can outscore a song that matches mood *and* energy perfectly. This
creates a "filter bubble": users who declare a genre rarely see cross-genre songs even
when those songs would objectively fit their vibe better.

**Dataset skew:** The 20-song catalog has only one track for several genres (punk, edm,
hip hop, folk, r&b). A user whose favorite genre is "folk" effectively gets one
guaranteed top hit and then falls back to mood/energy matches; there is no real
ranking *within* their taste.

**Silent failure on bad input:** A user asking for `mood: sad` gets no warning that
"sad" does not exist in the catalog; the mood term simply contributes 0 and the
recommendation silently ignores that preference (see the adversarial profile, where
"sad" was dropped and "pop + high energy" took over).

**Binary matching:** Genre and mood are strict equality checks — "indie pop" never
matches "pop", and "focused" never partially matches "chill", even though a human
listener would see the overlap.

---

## 7. Evaluation

I stress-tested the recommender with four user profiles:

1. **High-Energy Pop** — pop / happy / energy 0.85
2. **Chill Lofi** — lofi / chill / energy 0.35 / acoustic
3. **Deep Intense Rock** — rock / intense / energy 0.9
4. **Adversarial "Conflicted Listener"** — pop / sad / energy 0.9 / acoustic
   (mixes an upbeat genre with a sad mood that doesn't exist in the catalog)

**What matched intuition:** The Chill Lofi profile produced a clean, coherent top-5
(Library Rain, Midnight Coding, Study Fog, Focus Flow, Spacewalk Thoughts). Deep
Intense Rock correctly put Storm Runner on top by a wide margin.

**What surprised me:** "Gym Hero" keeps appearing near the top for any pop-leaning
or high-energy profile, even the adversarial sad one. It wins because it stacks a
genre bonus with a near-perfect energy match, which can overwhelm the fact that its
mood ("intense") has nothing to do with a "happy" or "sad" request. Similarly, the
adversarial profile still returned pop songs at the top, and the scorer silently dropped
the invalid "sad" mood rather than flagging the mismatch.

**Tests:** `tests/test_recommender.py` covers the loading and scoring math. Beyond
that, the profile comparisons above served as the main behavioral check.

---

## 8. Future Work

1. **Fuzzy matching for categorical fields:** Let "indie pop" partially match
   "pop", and let moods live on a similarity graph ("focused" ≈ "chill" ≈
   "relaxed") so the scorer stops treating near-matches as zero.
2. **Input validation:** Warn the user when a requested genre or mood doesn't
   exist in the catalog, instead of silently dropping it.
3. **Diversity penalty in top-K:** Penalize songs from an artist or genre
   already in the list so the top 5 isn't three LoRoom tracks in a row.
4. **Larger, better-balanced catalog:** Add more genres and moods (sad,
   romantic, nostalgic, classical, country) so a "folk fan" gets a real
   ranking within folk, not just "here's the one folk song we have."
5. **Weighted multi-preference profiles:** Let a user say "70% lofi, 30%
   jazz" instead of forcing one favorite genre.

---

## 9. Personal Reflection

**Biggest learning moment:** I assumed the weights on genre, mood, and
energy would all pull roughly equally on the rankings. Running the
weight-shift experiment showed me how wrong that intuition was. The top-1
pick was remarkably stable (no weight change moved it), while the middle of
the list was extremely sensitive. That changed how I think about "tuning" a
recommender: most of the action happens in ranks 2–10, not at the top.

**Where AI helped vs. where I had to double-check:** Claude was fastest at
mechanical scaffolding — expanding `main.py` to loop over profiles, drafting
the model-card section headers, even sanity-checking that my reverted
weights actually matched the originals. Where I had to slow down was
*interpreting* results: the AI happily produced plausible-sounding
explanations for rankings, and I had to verify those against the actual
score breakdowns before trusting them. The scoring math is simple enough
that anyone can check it by hand, and more than once that caught a
"plausible-but-wrong" claim.

**What surprised me:** How quickly four lines of arithmetic start to *feel*
like a recommendation. The explanations ("genre match (lofi) +2.0; mood
match (chill) +1.0; energy sim +1.00") read like a real product, even
though the logic is basically a weighted sum. It made me appreciate that a
lot of what we experience as "the algorithm" in Spotify or YouTube is less
magical than we think - it's just a scored ranking with more features and
much more data.

**What I'd try next:** Fuzzy genre matching and a diversity penalty in the
top-K. Both would address the two failure modes I saw most: "filter bubble"
effects from strict genre equality, and near-duplicate top lists dominated
by a single artist.
