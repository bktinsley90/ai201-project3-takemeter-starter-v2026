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

<!-- Baseline results from `baseline.py`. -->

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

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



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
