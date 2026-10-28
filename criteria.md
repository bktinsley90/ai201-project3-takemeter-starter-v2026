# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1.

Overall accuracy is at least 0.85 on the held-out test set, on every one of the three seeds (42, 7, 2024).

**Why this target:**

With two balanced labels a coin-flip guesser gets 50%, so 0.85 means it is wrong on fewer than 1 post in 6. I'm not aiming higher because I only have 250 posts (about 175 for training), and the source labels are noisy: some `suicide` posts never mention suicide, and the third-party posts are labelled inconsistently. The test split is only about 38 posts, so one post is worth about 2.6 points of accuracy; a seed-to-seed swing of 5 points is normal, and I'd call 0.85 a fair bar to clear each time.

---

## 2.

Both labels have an F1 of at least 0.80 on every seed: `suicide` >= 0.80 and `non-suicide` >= 0.80.

**Why this target:**

The labels are balanced (125 each), so there is no small label to excuse one side being much worse. About 19 test posts per label means F1 will move a few points between seeds, so I set this 5 points under my accuracy target instead of matching it. If the model gets 0.85 overall by being good at the everyday posts and bad at the suicide-related ones, criterion 1 would hide that and this one would catch it.

---

## 3.

Recall for `suicide` is at least 0.85 on every seed: of the real `suicide` posts in the test split, the model finds at least 85%.

**Why this target:**

In a tool that flags posts for a person to look at, missing a real `suicide` post is the worse mistake, and a false alarm is the cheaper one. So I want recall held higher than overall accuracy and I'm willing to accept lower precision. With about 19 `suicide` posts in test, 0.85 means missing at most 2. A recall of 0.80 would allow 4 misses, which I wouldn't call useful.

---

## 4.

When the model's confidence is 0.90 or higher, at least 0.90 of those predictions are correct, on every seed. At least 40% of test posts must reach that confidence level, so the criterion can't be met by the model never being confident.

**Why this target:**

A confident prediction should be right more often than the average one, otherwise the confidence score tells me nothing about which posts to trust. I set the bar at 0.90, above the 0.85 overall target, because that gap is the whole point of a confidence score. The 40% floor stops me passing by technicality when only 3 or 4 posts clear 0.90.

---

## 5.

My agreement with the staff set (`python agreement.py`) is at least 24 of 30 (80%).

**Why this target:**

I found in this dataset that the source labels don't consistently follow even my own rule (third-party posts go both ways). If I can't apply a written rule consistently, no model can learn it. 80% is deliberately above what I'd expect from a coin flip and below perfect: the staff set was chosen to include ambiguous posts, so I expect some honest disagreements. Falling below it would tell me my labelling is the problem, not the model. Note this one is measured once, not across seeds.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
