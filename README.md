# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---
## What This Does

This project trains a text classifier on Reddit-style posts from the public
Suicide Detection dataset (`data/Suicide_Detection.csv`; I use a 250-post
sample in `labels.csv`). It sorts each post into `suicide` or `non-suicide`.
The distinction I can point at is whether the writer says, about themselves,
that they want to die, want to kill themselves, or have a plan. These are the
dataset's labels, not a diagnosis or a judgment of anyone's immediate danger.

---

## Label Taxonomy

### `suicide`

**Definition:** The writer states their own wish to die, urge to kill
themselves, or plan or date for doing so.

**Example 1:**
> "badly want to kill myselfI've been thinking about it for months now and i
> guess im just saving up the courage to do it."

**Example 2:**
> "I plan on taking my life on 7/12/15, i'm tired of living."

### `non-suicide`

**Definition:** The writer makes no statement of their own suicidal wish,
urge, or plan; the post is about something else, or about someone else.

**Example 1:**
> "Time to clean my room 😔 Wish me luck, it's gonna take an effort. Be back
> in a bit"

**Example 2:**
> "Star Wars music somehow helps me focus when studying/doing homework"

### The hardest boundary

**Which two labels:** `suicide` and `non-suicide`

**The post that sits between them:**
> "My Friend Is Dying My friend swears she's going to kill herself before 21.
> Her mental state is terrible. ... I'm scared for her tbh."

It is full of suicide content, but the writer is describing a friend.

**The decision rule I used every time:** Label by what the writer says about
*themselves*. A first-person wish, urge, or plan is `suicide`. Suicide talk
about someone else (a friend, an ex, a family member) is `non-suicide`, however
serious.

**Where the source labels break this rule:** The dataset is not consistent here.
The friend post above is `non-suicide`, but "My friends mum is preventing him
from getting help ... edging closer and closer to suicide" is labelled
`suicide`. Some `suicide` posts also never mention suicide at all, e.g. "Does
anyone else think if you met yourself as a separate person, that you would kind
of hate yourself?" and "i need help just help me im crying so hard". Where the
source label disagrees with my rule, I'm recording it as label noise, not
correcting it silently.


---

## The Dataset


**Where the posts came from:** A random sample (seed 42) of 250 posts from the public Suicide Detection CSV (`data/Suicide_Detection.csv`, 232,074 posts, kept out of git because of its size). Only posts of 40–1,200 characters were eligible. The sample is saved as `labels.csv`.

**How I labelled them:** Mixed, and the `note` column in `labels.csv` says which is which. For 20 posts (10 per label, chosen at random) I read the post and labelled it myself, cold, before looking at the dataset's label; these are marked `cold`. I changed 2 of those 20 away from the dataset's label, both from `suicide` to `non-suicide`, because the writer was not stating a wish to die under my first-person rule. The other 230 posts keep the dataset's original `suicide` / `non-suicide` label, which I treated as a pre-label and have not re-read; these are marked `pre-label`. So most labels here are the source's, not mine, and the cold set is the only part I can vouch for.

**Counts per label:**

| Label | Count | Share |
|---|---|---|
| `suicide` | 123 | 49.2% |
| `non-suicide` | 127 | 50.8% |
| **Total** | 250 | 100% |

**Three hard cases**

**1.**
> *The post:* "I hurt so much. And I'm not allowed to show it. ... My chest physically hurts from how unhappy I am. I've cried myself to sleep almost every night for the past ten years. Im so tired of it all. ... I just want it to be over. I want to go away and sleep. Help me. Please help me."
>
> *Could have been:* `suicide` (the source label) or `non-suicide`.
>
> *I chose:* `non-suicide`, because the post never says the writer wants to die or to kill themselves. "Over" and "go away and sleep" could mean that, or just wanting the exhaustion to stop. My first-person rule needs an explicit wish, urge or plan, and this has none. It is the closest call of the three, since the wording can be read either way.

**2.**
> *The post:* "I feel so useless. I just feel like nobody likes me. I've never felt this suicidal in several months. Idk what's happened. All I have is my service dog. Still, I feel no point in life. No future. It can only get worse from here. ... I just want to vent in real life."
>
> *Could have been:* `suicide` (the source label) or `non-suicide`.
>
> *I chose:* `non-suicide`, because "suicidal" appears only as a comparison with earlier months, and nothing says the writer wants to die or has a plan. The word is the cue a model is most likely to latch onto, which is why I paused. The rule looks at what the writer says they want, not at which words appear.

**3.**
> *The post:* "My Friend Is Dying. My friend swears she's going to kill herself before 21. ... She says she may start cutting herself soon. ... I'm wondering if I should send screenshots of our conversations to her mother ... I don't know what to do at this point."
>
> *Could have been:* `suicide` or `non-suicide`.
>
> *I chose:* `non-suicide`, which matches the source label. It is full of suicide content but all of it is about the friend; the writer states no wish of their own. The dataset is inconsistent on this: "My friends mum is preventing him from getting help", also about a friend, is labelled `suicide`, and I have not changed it.

---

## The Training Run

<!-- Your starting model, your settings, and anything you changed and why. -->

**Base model:** `distilbert-base-uncased`, a small pretrained English text model that I fine-tune to sort posts into `suicide` and `non-suicide`.

**Settings:** 3 epochs, learning rate 2e-5, batch size 16, seed 42, max length 128 tokens. Trained on Apple Silicon GPU (MPS).

**Anything I changed from the defaults, and why:** Nothing. I used the notebook's defaults as given. The max length of 128 tokens means longer posts are cut off, so the model may not see the end of them.

**Split sizes:** The notebook splits 70/15/15, stratified by label: 174 train / 38 validation / 38 test (250 posts). The test split has 19 `suicide` and 19 `non-suicide`. Both labels are above the roughly 8-per-label warning level, but 19 is still small: one post is about 2.6 points of accuracy (5 points of per-label recall), so scores will move between seeds.

---

## How I Used AI

**Moment 1: the Training Run section**

- *What I asked for:* I asked the AI to fill in the Training Run section and told it I had not changed any defaults.
- *What came back:* It read the settings from `takemeter.ipynb` (`distilbert-base-uncased`, 3 epochs, learning rate 2e-5, batch size 16, seed 42, max length 128) and the test split from `test_split.csv` and `results.json` (38 posts, 19 per label). It also noted that posts over 128 tokens are cut off, and that 19 test posts per label is small enough for scores to jump between seeds.
- *What I changed:* I checked the settings against my own run. The 174 / 38 train and validation sizes were worked out by the AI from the notebook's 70/15/15 rule, not read from a run output, so I confirmed them against what my notebook printed.

**Moment 2: the Label Taxonomy**

- *What I asked for:* I asked the AI to double-check my Label Taxonomy against the assignment's description of a strong taxonomy.
- *What came back:* It found that my definitions were circular ("a post the dataset labels X"), that one `suicide` example had no suicide content in it, and that my decision rule contradicted the data (a post about a friend's suicide threat was labelled `non-suicide`). It rewrote the taxonomy around one checkable rule: label by what the writer says about *themselves*. It also flagged where the source labels break that rule.
- *What I changed:* I reviewed the rewritten rule against 20 posts I labelled cold, and changed 2 of the source labels from `suicide` to `non-suicide` under it. Both are written up in the hard cases.

**Pre-labelling disclosure:** Yes, most of the dataset was pre-labelled. The AI drew a random sample of 250 posts (seed 42) from the Suicide Detection CSV and wrote them to `labels.csv` with the source dataset's labels already attached. I labelled 20 of them cold (10 per label) and changed 2 of those. The other 230 are marked `pre-label` in the `note` column. I have not re-read those 230, so they still carry the source dataset's labels, including the inconsistencies described in the taxonomy.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

| Measure | Baseline |
|---|---|
| Overall accuracy | 0.868 |
| Macro F1 | 0.866 |
| F1 — `non-suicide` | 0.848 |
| F1 — `suicide` | 0.884 |

**What I predicted before I looked:**
I predict the trained model will beat the baseline on F1 for `suicide`, because fine-tuning should help it distinguish first-person suicidal statements from posts that discuss suicide without expressing the writer's own intent.

**What the gap actually means:**
On the same 38 held-out posts, the trained model scored 0.895 accuracy and
0.894 macro F1, slightly above the definition-based baseline (0.868 and 0.866).
Its F1 improved for both `non-suicide` (0.889 vs. 0.848) and `suicide` (0.900
vs. 0.884). This is a modest improvement on this small test set, not proof that
fine-tuning will generalize better; the earlier comparison also showed that
using my written definitions raised baseline accuracy by 21.1 percentage
points over bare label names.

---

## Run Log — Before

Device: Apple Silicon GPU (MPS), torch 2.14.1, max length 128. Each seed was
scored on its own 38-post test split (19 per label).

| Measure | Seed 42 | Seed 7 | Seed 2024 | Spread |
|---|---|---|---|---|
| Overall accuracy | 0.895 | 0.737 | 0.842 | 0.158 |
| Macro F1 | 0.894 | 0.736 | 0.840 | 0.158 |
| F1 — `suicide` | 0.900 | 0.750 | 0.857 | 0.150 |
| F1 — `non-suicide` | 0.889 | 0.722 | 0.824 | 0.167 |

Accuracy moved 0.158 across seeds, more than 10 points. My dataset is too
small for a stable measure: 250 posts leaves a 38-post test split, so one post
is worth about 2.6 points of accuracy and the split a seed happens to draw
changes the score a lot. The labels are balanced (127 / 123), so lopsidedness
is not the cause. The diagnosis is size, plus noisy source labels. Seed 42's
0.895 was a friendly draw, not the number to trust.

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | >= 0.85 on every seed | 0.895 | 0.737 | 0.842 | MISSED (seeds 7 and 2024) |
| 2. Per-label F1 | both labels >= 0.80 on every seed | 0.900 / 0.889 | 0.750 / 0.722 | 0.857 / 0.824 | MISSED (seed 7) |
| 3. `suicide` recall | >= 0.85 on every seed | 0.947 | 0.789 | 0.947 | MISSED (seed 7) |
| 4. Confidence | >= 0.90 right at confidence >= 0.90, on >= 40% of posts | not measured | not measured | not measured | NOT YET MEASURED |
| 5. Agreement with staff set | >= 24 / 30 | not run | not run | not run | NOT YET MEASURED |

Criterion 4 needs per-post confidence at the 0.90 threshold, which
`results_three_seeds_before.json` does not store; it only has thirds. The mean
confidence on correct predictions was about 0.67 on every seed, so few posts
likely reach 0.90. Criterion 5 is measured once with `agreement.py`, not per
seed.

### Confusion matrix

Seed 42 (38 test posts):

| true \ predicted | suicide | non-suicide |
|---|---|---|
| **suicide** | 18 | 1 |
| **non-suicide** | 3 | 16 |

**My biggest off-diagonal number, and what it means:**
3 real `non-suicide` posts were called `suicide`, and only 1 real `suicide`
post was called `non-suicide`. The model leans toward `suicide` on this pair:
it has picked up that suicide-related content means `suicide`, but not my rule
that the writer must be talking about themselves, so posts about a friend or
someone else are the ones it gets wrong. The lean holds on the other seeds too
(summed over all three seeds, 14 `non-suicide` posts were called `suicide`
against 6 the other way). A lean in one direction can point at my labelling
rather than the model, since the source labels treat third-party posts
inconsistently. I can't separate the two until I run the agreement report.



---

## Verdicts and Diagnoses

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | Overall accuracy | >= 0.85 on every seed | MISSED | 0.895, 0.737, 0.842. Seed 42 clears it; seeds 7 and 2024 do not. |
| 2 | Per-label F1 | both labels >= 0.80 on every seed | MISSED | Seed 7 is 0.750 (`suicide`) and 0.722 (`non-suicide`). Seed 2024's `non-suicide` is 0.824, so only seed 7 fails. |
| 3 | `suicide` recall | >= 0.85 on every seed | MISSED | 0.947, 0.789, 0.947. Seed 7 missed 4 of 19 real `suicide` posts, against a limit of 2. |
| 4 | Confidence | >= 0.90 correct at confidence >= 0.90, on >= 40% of posts | NOT MEASURED | `results_three_seeds_before.json` stores accuracy by confidence third, not the share of posts at 0.90 or above. Mean confidence on correct predictions was only 0.67-0.68 on every seed, so a 40% share at 0.90 looks unlikely, but I haven't measured it. |
| 5 | Agreement with staff set | >= 24 / 30 | MET | 30 / 30 (100%). Measured once, not per seed. |

**Diagnoses**

**Criteria 1, 2 and 3: too few examples.** The three misses share one cause,
and the evidence is the spread. Accuracy moved 0.158 across seeds on the same
data and settings, and the only thing that changed was which 38 posts landed
in the test split. With 250 posts (174 for training), one test post is worth
about 2.6 points of accuracy and about 5 points of recall. Seed 7 dropped
recall from 0.947 to 0.789 by missing 3 more `suicide` posts, which is the
size of change a small, hard-to-predict test split produces. The labels are
balanced (127 / 123), so this is not a lopsided-label problem.

**Criteria 1 and 2: a hard label pair, mostly one direction.** Where the model
fails it leans toward `suicide`: on seed 42 it called 3 real `non-suicide`
posts `suicide` and missed 1 the other way, and on seed 7 it was 6 against 4.
The hard cases are posts full of suicide content that aren't about the writer
(a friend, an ex), which my decision rule says are `non-suicide`. I can't tell
from the saved files which posts seed 7 got wrong, so I haven't confirmed
that these are the ones it missed.

**What the agreement report tells me about labelling versus the model.** I
matched staff on 30 / 30, so I can apply a written rule I didn't write
consistently. That points away from sloppy application on my side. It does not
clear the training labels: 230 of the 250 are source-dataset labels I haven't
re-read, and I already found the source labels treat third-party posts
inconsistently (the friend post is `non-suicide`, "My friends mum is
preventing him from getting help" is `suicide`). So an inconsistent boundary in
the data is still a live cause for the over-calling of `suicide`. The
agreement report can't separate that from the model finding the pair hard.
Re-reading the third-party posts among the 230 pre-labels is the test that
would.

**Criterion 4:** not diagnosed, because it hasn't been measured.

---

## Agreement Report

**Agreement rate:** 30 / 30 = 100%

I matched staff on every post, so there are no disagreements to adjudicate.
Results are in `agreement_results.json`.

**Disagreements**

None: 0 of 30. Nothing falls into pile (a), (b) or (c).

**What the pattern in my disagreements tells me:**
With no disagreements there is no pattern to read, so I can only say what the
100% does and doesn't show. It does show I applied a rule I didn't write the
same way its authors did across all 30 posts, and that rules 1 to 3 settle the
hardest boundaries (sp_014's decorative 3% as `hot_take`, sp_010's argument
from a measured gap as `analysis`) without needing the rule 4 tie-break. It
does not prove my rule is genuinely tight. The staff taxonomy says several of
these 30 were chosen as ambiguous, and a rate in the 70s was expected, so
getting all 30 may mean the set was easier than advertised: it splits
cleanly into three blocks of ten (10 per label), which may have made it easier.
The honest reading is that I can follow a written decision rule consistently,
which is good evidence for my own `suicide` / `non-suicide` rule being
applicable the same way, but 30 posts and one rater is too little to call it
tight.

---

## The Improvement

**What I changed:** I added 30 `non-suicide` posts to `labels.csv` (250 → 280 rows; 157 `non-suicide` / 123 `suicide`). `non-suicide` had the worst F1 on all three seeds (0.889, 0.722, 0.824 vs 0.900, 0.750, 0.857 for `suicide`), so it is the label I treated as thinnest. By raw count it is not thinner (127 vs 123), so this was chosen on F1, not on count. About 18 of the 30 are boundary cases under my rule: the writer talks about a friend, sibling, family member, ex or a public figure's suicide or self-harm, or quotes it jokingly, but says nothing of their own wish or plan. The other 12 are ordinary or off-topic posts. The AI picked them from the unused part of `data/Suicide_Detection.csv` (source label `non-suicide`) and read each one against the rule. I did not hand-label them; their `note` column says `added-improvement (AI-labelled under README rule)`. The set is not balanced any more, which I will watch in the After run.

**Which diagnosis pointed at it:** Two from my Verdicts and Diagnoses: too few examples for the label that does worst, and a hard label pair where the model leans toward `suicide` (3 real `non-suicide` posts called `suicide` against 1 the other way on seed 42, 14 against 6 summed over seeds). Posts about someone else's suicide are the likeliest place for that lean, so I added them on purpose. I have not confirmed this, because the per-post errors were not saved.

### Run Log — After

Same device and settings as Before. Each seed was scored on its own 42-post test split (18 `suicide`, 24 `non-suicide`) from the 280-post set, so it is not the same test set as Before. One post is about 2.4 points of accuracy. Results are in `results_three_seeds_after.json`.

| Measure | Seed 42 | Seed 7 | Seed 2024 | Spread |
|---|---|---|---|---|
| Overall accuracy | 0.976 | 0.810 | 0.786 | 0.190 |
| Macro F1 | 0.976 | 0.808 | 0.780 | 0.196 |
| F1 — `suicide` | 0.973 | 0.789 | 0.743 | 0.230 |
| F1 — `non-suicide` | 0.979 | 0.826 | 0.816 | 0.162 |

Accuracy moved 0.190 across seeds, more than 10 points, and the spread got wider than Before (0.158). The diagnosis is the same: the dataset is still too small for a stable measure. 280 posts leaves a 42-post test split, and which posts a seed draws matters more than the training change. The labels are now less balanced (157 / 123), which I made worse on purpose.

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | >= 0.85 on every seed | 0.976 | 0.810 | 0.786 | MISSED (seeds 7 and 2024) |
| 2. Per-label F1 | both labels >= 0.80 on every seed | 0.973 / 0.979 | 0.789 / 0.826 | 0.743 / 0.816 | MISSED (seeds 7 and 2024, on `suicide`) |
| 3. `suicide` recall | >= 0.85 on every seed | 1.000 | 0.833 | 0.722 | MISSED (seeds 7 and 2024) |
| 4. Confidence | >= 0.90 right at confidence >= 0.90, on >= 40% of posts | not measured | not measured | not measured | NOT YET MEASURED |
| 5. Agreement with staff set | >= 24 / 30 | 30 / 30 (one run, not per seed) | | | MET |

**Did it help, and how do I know:** Partly, and I can't call it proven. What improved is the label I added to: `non-suicide` F1 went up on seed 42 (0.889 to 0.979) and seed 7 (0.722 to 0.826) and was flat on seed 2024 (0.824 to 0.816), so the mean went from 0.812 to 0.874. The lean toward `suicide` shrank: real `non-suicide` posts called `suicide` fell from 3 to 1 on seed 42, and summed over seeds there are now 10 of those against 8 the other way, where Before it was 14 against 6. Mean accuracy rose from 0.825 to 0.857.

What got worse: `suicide` recall fell from a mean of 0.894 to 0.852, and on seed 2024 it is 0.722 (5 of 18 `suicide` posts missed). The `suicide` F1 mean is unchanged (0.836 to 0.835). Seed 2024's accuracy dropped from 0.842 to 0.786. The model seems to have moved its boundary toward `non-suicide` rather than learning the third-party distinction cleanly. That fits adding 30 examples to only one label.

Why I can't call it a win: the verdicts on criteria 1, 2 and 3 are the same as Before (MISSED on the weak seeds). The test split changed with the dataset, so Before and After aren't scored on the same posts. The seed-to-seed swing (0.19) is larger than the gains I'm pointing at. And the 30 added posts were labelled by the AI under my rule, not by me. The one clear result is that the extra `non-suicide` posts reduced the over-calling of `suicide` and cost some `suicide` recall.

---

## What's Still Broken

Criteria 1, 2 and 3 are still MISSED after the improvement. Criterion 4 was never measured. Criterion 5 is met.

| Criterion | Still missed on | What I would do | Why I stopped |
|---|---|---|---|
| 1. Accuracy >= 0.85 | Seeds 7 (0.810) and 2024 (0.786) | Label more posts to get a bigger test split, and report the mean over seeds as well as the worst seed | A 42-post test split moves 2.4 points per post and the seed swing is 0.19, so more tuning would only chase noise |
| 2. Per-label F1 >= 0.80 | `suicide` on seeds 7 (0.789) and 2024 (0.743) | Add `suicide` posts too, not only `non-suicide`, so the boundary does not drift the other way | My one improvement added only `non-suicide` posts, and that is what cost `suicide` recall |
| 3. `suicide` recall >= 0.85 | Seeds 7 (0.833) and 2024 (0.722) | Same as criterion 2, then re-check recall | It got worse after the change (mean 0.894 to 0.852), so I would not keep tuning on `non-suicide` |
| 4. Confidence | Not measured | Write a retraining script that saves per-post confidence, then count posts at >= 0.90 | The saved JSON only has confidence thirds, so I could not measure it without retraining |

The cause of the remaining misses is the one I diagnosed earlier: too few examples and a hard label pair, plus labels I have not reviewed (230 of the first 250 are unreviewed source labels, and the 30 I added were labelled by the AI). I cannot separate a labelling problem from a model problem until I re-read those labels and save which posts each seed gets wrong.

**The gap between what I meant my labels to capture and what the model
learned:**
I meant `suicide` to mean the writer is talking about their own wish, urge or plan, but after the improvement the model still misses real `suicide` posts (seed 7: 3 of 18, seed 2024: 5 of 18) and still calls some `non-suicide` posts `suicide` (5 on seed 7, 4 on seed 2024). So it learned something close to my rule, but not the rule: on the weak seeds it is wrong in both directions, which points at inconsistent training labels as much as at too little data.

<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
