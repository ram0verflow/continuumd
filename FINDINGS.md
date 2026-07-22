# continuum: research log

A chronological lab notebook of what was tried, what held, and what was retracted. Ordering is by when the work happened, not by how it reads; where an earlier confident claim was later overturned, the retraction is left visible next to the original rather than edited away. The measured results that survive are summarized in [README.md](README.md).

## How this was built

I started from an older Python prototype and a short spec. The port to Rust gave me a four level store (identity, topic summaries, details, raw archive) where every value keeps its version history, and a kernel/driver split where the kernel is domain agnostic and each driver owns retrieval for one kind of memory. There is a conversation driver and a code driver.

Retrieval took the longest to get right. The first version walked a topic tree down a single path and missed most things. Beam search over the tree helped. Adding BM25 alongside embeddings helped recall but made answers worse, and it took me a while to see why: I was loading around 100 messages per query and the 8B model would miss facts that were sitting in plain sight in its own context. Capping the load at 30 messages, reranked and then presented in chronological order, fixed accuracy and halved latency at the same time.

Dates were the worst question category. LoCoMo gold answers look like "the week before 27 June 2023" and a small model is bad at calendar math. So the code resolves phrases like "last week" or "yesterday" against the timestamp of the message that said them, in plain Rust, and injects the resolved date as a note. That category went from 62% to 81% on my local judge.

The fine tune came from the benchmark itself. Conversations 1 through 9 supplied synthetic training examples of three kinds: evidence loaded so answer it, evidence withheld so say CONTEXT_NEEDED, and a trap loaded (a question about the wrong person) so still say CONTEXT_NEEDED. Conversation 0 was held out and never used for training. QLoRA on llama 3.1 8B through MLX, overnight on the laptop. The first round refused too much. The second round changed the mix and paired every trap with a question the same context could answer. The share of refusal examples in the training data acts like a dial for how conservative the model is.

Two things about evaluation that I got wrong at first and had to fix. I was grading answers with the same 8B model that produced them, and when I re graded with Claude Haiku the score dropped 19 points. Every number below is from the external judge. And I worried the base model might know this public dataset from pretraining, so I asked it the conv 0 questions with no memory attached. It scored 1.3%, which is guessing.

## Ablations, other models, a second benchmark

Removing one retrieval component at a time (conversation 0, base llama 3.1,
ROUGE-L, same 154 questions):

| configuration | ROUGE-L | page faults |
|---|---|---|
| full pipeline | 0.449 | 14 |
| without tree routing | 0.455 | 12 |
| without dense embeddings | 0.442 | 16 |
| without the date resolver | 0.424 | 13 |
| without BM25 | 0.265 | 66 |
| without the 30 message cap | 0.118 | 1 |

The cap and BM25 carry the system. Removing the cap reproduces the failure
that shaped the design: the model reads a hundred loosely relevant messages,
answers wrongly with confidence, and generation time doubles. The tree adds
nothing on this benchmark; its case is browsing and scale, not QA, and I keep
it because the online ingestion path builds it for free.

Same stack, different answer models, same questions, graded by claude haiku:
llama 3.1 8b 48.0%, mistral 7b 45.4%, phi3 mini 50.0%. Three model families
land in the same band on the identical stack. ROUGE had suggested phi3 was
far worse (0.215 against llama's 0.449); the judge shows it is just verbose,
not wrong. Refusal discipline is where models differ wildly: llama refuses
25.5% of unanswerable questions, mistral 28.9%, phi3 almost never (2.2%),
which is the behavior the fine tune exists to install.

I also tried a second retrieval hop: mine the first round's results for
names the query did not contain, search again on those, merge at a discount.
It moved the chained fact tasks a little (qa2 went 0/20 to 4/20) but cost
more than it paid: qa1 dropped 13/20 to 8/20 from the added noise and
latency doubled. It ships default off behind a flag. The honest conclusion
is that blind expansion is the wrong shape for multi hop here; the kernel
already has a targeted mechanism (the model faults, the kernel re-pages on
the fault topic) and that is the direction worth pursuing.

I tested whether the fine tuned model would trigger that mechanism on the
chained tasks. It does not: zero faults fired, because the loaded chunks
look topical even when the reasoning chain through them is incomplete. The
tune teaches "fault when the topic is missing," not "fault when a chain is
missing a link," and those are different skills. Teaching the second one
needs training examples of exactly that shape, which are easy to generate,
and is the obvious next round if I train again. The tuned model also scored
ten points below base on this benchmark overall; it is better on
conversational memory and worse on book text, so different memory volumes
probably want different answer models, which the driver design happens to
support.

Retrieval cost, measured: 33 to 57 ms at the median against 7 to 14 seconds
of generation. The memory side is about half a percent of a query.

BABILong (facts hidden in 64k tokens of book text, answered through a 4k
window, exact match): qa1 13/20, qa2 0/20, qa3 2/20, qa4 12/20, qa5 20/20.
Single fact tasks work well through sparse retrieval. Chained fact tasks
fail, the same multi hop weakness LoCoMo showed, and the known failure mode
of retrieval systems generally. Fetch the data with `fetch_babilong.py`,
run with `cargo run --release --bin babilong`.

There is an endurance script that hammers the live loop: it plants ten
facts, buries them under a hundred turns of unrelated chatter on the small
fixed window, then asks for them back. 130 turns over 104 minutes, the
window never went over budget, 9 of 10 facts came back (the tenth was a
grader casing bug). `python3 endurance.py <port>` against a running server.

The daemon has its own compaction stress harness. It forces the session
window down to 500 tokens, plants ten facts, buries them under thirty
distractor turns so every planted fact gets demoted out of the window,
contradicts two of them along the way (the dentist moves, the rate limit
drops), then asks for everything back plus two things that were never
said:

```
CONTINUUM_HOME=/tmp/continuum-stress ./target/release/continuumd --port 4311
python3 stress_daemon.py 4311
```

The harness grades three claims separately, because they are different
claims. Two runs at n of 10 so far, hosted answer model (Nova 2 Lite)
over the local memory stack. The first run scored clean across the board;
the second disagreed with it in both directions, which is exactly why
these stay labeled a smoke test and not a benchmark. The LoCoMo numbers
above are the measured ones.

Never-said probes, and this is the result I care most about. Asked for a
pool locker combination when only a gym one exists, and for a wedding
date never mentioned, run one page faulted both times and answered an
honest "I don't have that". Then, in one of two adversarial trials, the
model answered a question about a locker that was never mentioned by
returning the gym combination with full confidence. I am deliberately not
attaching a ratio to that: a ratio invites averaging, and this is the
confabulation the probe exists to catch, the failure mode that erodes
trust fastest, and the argument for the fault fine-tune in the answer
model rather than a solved problem.

Retrieval survives window churn: 19 of 20 across the two runs, one stale.
The window peaked around 499 of 500 and never went over, about 85
messages per run were demoted to the archive, and facts came back exact
at 32 to 59 ms retrieval. The contradicted facts usually answered with
the new values (October 21st, 90 per minute) but not always: run two
answered the API question with the superseded 120. To be precise about
attribution: in the daemon's flow the store never enters the prompt
(identity aside), so this recall is served by the conversation index that
every turn feeds. The store's runtime jobs are identity, provenance, the
archive, and the browser.

Write-back capture, measured on its own, and split by where the fact
landed, because a fact that only survives inside the identity string has
weaker guarantees than a versioned branch detail. Run one: 3 branch-filed,
4 identity-only, 3 missed. The identity blob had absorbed the locker
combination and the database version as if they were who I am. That
prompted three deterministic kernel fixes: value-bearing facts reroute
from identity to a branch detail, near-duplicate identity merges are
blocked by token overlap, and a detail that narrowly restates a branch
summary supersedes it, so a summary can't keep advertising the dentist's
old date after the details learn the new one (verified in the store:
the summary now versions 14th to 21st with history kept). Run two, with
the fixes: 5 branch-filed, 1 identity-only, 4 missed. The misses that
held across both runs are events involving another person (lending a
book) and corrections phrased as "we lowered X to 90"; the other misses
flip run to run and look like noise. The classifier keeps what sounds
like a profile statement and drops events and corrections, which is
actionable in a way "capture is weak" is not. The harness tags every
planted fact by form and reports branch versus identity placement, so
further runs show whether the pattern holds. Capture is also not what
serves recall today, which is worth knowing before reading these numbers
as failures.

Which raised the question the store had coming: does any of that capture
machinery matter to the thing that ships? I wired query-relevant store
topics (summary plus current fact values) into the prompt behind a
setting, instrumented every answer with how many topics got paged in, and
ran the harness a third time. The answer, on this evidence, is no. Recall
was 10 of 10, but the one question where the store could have earned its
place (the corrected rate limit, which a previous run answered stale) was
paged zero store topics, because the correction was never captured; the
pass came from the driver index having a better day. Only four of ten
questions pulled in any store topic, and all four also had the raw
messages available. One real if narrow win: the dentist question paged in
the superseded summary and answered the new date, which is the case the
stale-summary fix exists for; before that fix this experiment would have
pushed the old date into the prompt. The pool-locker probe leaked again
in this run, its second leak in three trials, with and without the store
block, so the block neither causes nor prevents that. Net: store context
ships default off, the toggle stays, and the store's runtime jobs remain
identity, provenance, the archive, and the browser. If capture gets good
enough to know things the raw history doesn't say plainly, this
experiment is sitting there ready to rerun.

That verdict had a hole in it, though: every fact in the harness lives in
one message, which is the case raw retrieval was never going to lose. So
there are two more instruments. `leak_gauntlet.py` measures near miss
confabulation per answer model: plant a gym locker combination, then ask
about a pool locker, a bike lock, an office locker, and a train time that
were never mentioned, twice each, on a fresh state per model. Nova Pro
leaked 0 of 8. Llama 3.3 70B leaked 1 of 8, but two of its trials were
eaten by rate limits so its denominator is soft. Nova 2 Lite and qwen2.5
14B leaked 2 of 8 each. Every single leak, on every model that leaked,
was the pool locker question specifically; the other three frames never
tempted anyone. The daemon now defaults to the model that never leaked,
and the harder tests below ran on it.

`stress_discriminate.py` plants cases that do not live in one message:
synthesis (the dentist date in one turn, "pushing everything in my
calendar back a week" thirty turns later, phrased to share no keyword
with the question), contradiction chains (an editor changed three times,
a standup time changed twice, every mention keyword-equal to the
question so recency cannot come from lexical luck), and cross topic
composition (a 50 thousand request plan and a 62 thousand usage stated
separately, then "am I over my allowance"). Run twice on the clean
model, store context off and on, fresh state each.

Both conditions scored 4 of 5, and the details matter more than the tie.
Raw retrieval plus an answer model that combines at answer time handled
synthesis in both conditions: the off condition reply literally walked
"originally October 14th, pushed a week, October 21st", and both
contradiction chains resolved to the latest value. "Search cannot
combine" undersells what the answer model does with a well ordered
working set, at least at this transcript size, and that is a real
revision to my expectations, which had the store winning these. And both
conditions failed the same case, the composition one: retrieval surfaced
the usage but not the plan (the question shares keywords with only one
of them), and the store had not captured both facts either, so the block
paged in nothing. Neither architecture can do this today. It is the same
shape as the chained fact weakness LoCoMo and BABILong already showed,
and the fix candidates are the same: fault driven second retrieval, or
capture reliable enough to pre join facts that belong together.

The honest limits: one run per condition, and a 34 turn transcript, which
with a 30 message load cap means retrieval had good odds of surfacing
scattered mentions by volume alone. Whether answer time synthesis
survives a transcript of hundreds of messages is the open question, and
the scale run that answers it needs the harnesses above run dozens of
times per condition, which is hours of model time and the next piece of
work on this thread.

The composition failure got its fix, and the fix got a regression test,
and the regression test caught two of my own bugs before it caught the
system improving. Memory faults now chain: the fault loop handles
CONTEXT_NEEDED alongside web and tools, so a model holding half an
answer can fault for exactly the missing counterpart and get one
targeted re-page, with a dedup set (the same token overlap comparator
the identity guard uses) stopping a chain from asking the same thing
twice in different words. Arithmetic on remembered values goes through a
deterministic evaluator now, not model math: the model raises
CALC_NEEDED, a small parser computes sums and date shifts exactly, and
the result comes back as context. A single regression run went 5 of 5,
and the journal showed the machinery earning it: October 14 plus 7 days
and 62000 minus 50000 both went through the calculator. Then I ran it
seven more times, and the single run turned out to be the outlier.

Here is the honest scale result (Nova Pro, seven valid runs across short
and long transcripts; one eighth run was a Bedrock transport failure and
is excluded, not scored as zero). Adjudicated by hand from the reply
text, because the automated grader was still false-passing, of which
more below:

- Arithmetic through the calculator: 7 of 7. When both operands are in
  context, the deterministic evaluator is reliable, and this is the one
  clean win of the round.
- Contradiction chains (editor changed three times, standup twice): 7 of
  7. Chronological ordering carries recency; the model picks the last
  value every time.
- Date-shift synthesis: 3 of 7, four stale. The second mention ("pushing
  everything back a week") shares no keyword with "dentist", so at scale
  retrieval usually does not surface it, the model never sees a reason to
  compute anything, and it answers the raw October 14th. This is the
  composition failure wearing a different hat, and it confirms the
  prediction that answer-time synthesis is the first thing to degrade at
  length.
- Composition (am I over my allowance): 1 of 7. The fault chain that was
  the whole point of the fix mostly does not fire. Six of seven times the
  model answered "to determine that, I need to know your usage" and asked
  the user for a fact that was sitting in memory, instead of raising
  CONTEXT_NEEDED for the missing plan size. The one pass was a long
  transcript with store context on, where the store block happened to
  carry the plan size into the prompt directly.

So the composition fix did not work, and I am leaving that thread open.
The 5 of 5 regression was retrieval luck on a single run. Two process
notes, because they are the point as much as the numbers are. The
automated grader false-passed the composition case in five of these
runs: the needle "you're over" matched the model's own "if you're over
your monthly allowance", the third time this exact question-echo bug has
appeared in this test and the second time I thought I had killed it. The
needle is now verdict-only. And the lone bright spot is a real one worth
keeping: that single composition pass came from store context, the
feature I had measured as useless and shipped default off. For a fact
that retrieval structurally cannot reach, the store block put it in front
of the model when nothing else would. That is n of 1 and not enough to
turn the setting back on, but it is the first evidence in this whole
thread that the store's runtime path might earn its place on exactly the
case the driver index cannot serve. The next real fix is making the
fault chain actually fire when the model holds half an answer, which the
model mostly will not do on instruction alone.

That last sentence turned out to be wrong in an encouraging direction:
the model will do it on instruction, if the instruction shows it how.
Before committing to a fine tune, the fault instruction was rewritten to
be example driven, naming the competing habit outright ("never ask the
user to supply a fact, faulting IS how you look it up") with three
worked cases of holding half an answer and faulting for the named gap.
Same seven run protocol, only that change: composition went from 1 of 7
to 4 clean passes of 7, and the fault fired in five of seven runs after
never having fired at all. The journals show the whole chain doing its
job for the first time: CONTEXT_NEEDED for the plan limit, one targeted
re-page, 62000 minus 50000 through the calculator, verdict. The costs,
also measured: two runs wedged into the fallback voice after an
unsatisfiable protocol line, one long transcript run got the direction
right but garbled the magnitude ("exceeded by 62,000 calls"), the model
now occasionally faults on general knowledge questions it should just
answer (twice in seven runs, on a warmup question), and the date shift
case did not move, because a model that thinks it has a complete answer
has no reason to fault. Instruction fixes the known-gap case; it cannot
fix the silent-staleness case. The fine tune stays on the shelf unless
that second shape starts to matter more.

The store context follow up got its replication test in the same sweep:
eight fresh composition cases, each verified programmatically so the
question cannot lexically reach the second fact, store context on
against off, per answer attribution. Result: a dead tie, five of eight
each (after the grader produced its first false NEGATIVE, a correct
"you are over the engineering hiring budget" scored as a miss for not
containing the needles; the fixture suite now covers that direction
too). The n of 1 signal did not replicate, and the composition passes in
both conditions were fault-and-calculator work with zero store topics
paged. So store context stays off and the graph question loses its
best-looking evidence. But the sweep bought something better than the
answer it was designed for, in one failure trace: the model faulted
correctly for "total number of engineers hired this year", and the
re-page found nothing, because the stored fact says "15 people on the
platform team", which shares not one word with the model's phrasing of
the gap. The fault chain names the missing thing in the model's
vocabulary and searches in the user's. Retrieval by shared words fails
across that gap in both directions, and no amount of prompting fixes
it. That, not the store, is now the standing argument for an entity
level reachability path. One more hazard from the same sweep, present
with the store on and off alike: cases that share unit words bleed into
each other, one answer subtracted the storage tier's 140 gigabytes from
the photo library's 620 because both mention gigabytes. Reachability by
vocabulary is too blunt in both directions at once, it misses what is
phrased differently and conflates what is phrased alike.

So the cheapest counter to that finding got built and measured before
any graph work: fault re-pages can now union in the fault topic's pure
dense neighbours, candidate gate bypassed, behind a setting. The trace
that motivated it flipped on the first traced run: the model faulted for
"current number of engineers hired this year", the semantic re-page
carried "15 people on the platform team" into context, the calculator
did 15 minus 12, and the answer was a clean "over the engineering hiring
budget by 3 engineers", in both store conditions. Vocabulary mismatch on
the fault path is bridgeable at retrieval time. What the same sweep gave
back with the other hand: the store on/off tie still did not break
(hand adjudicated 6 of 8 against 5 of 8, store off ahead by noise), and
the conflation got worse where it was already worst. Both conditions now
import the storage tier's 140 gigabytes as the drive's current usage and
conclude a 620 gigabyte library fits a 500 gigabyte drive with both
correct numbers present in the same sentence. The wider net bridges gaps
and feeds conflation with the same motion. That sharpens what a graph
would actually be for: not recall, which expansion now covers, but
precision, keeping facts attached to their entities so a number cannot
drift between cases just because both say gigabytes.

Housekeeping from the same round, briefly. The wedge from the previous
sweep was diagnosed by the first fully traced run: the model burned
three of four action loop rounds on calculator syntax the parser
refused, "15 (current headcount) - 12 (budgeted headcount)", then the
word minus, then unit words. The calculator now normalizes model shaped
expressions, those three are regression tests, and the traced sweeps
since show zero wedges in sixteen questions. The fault dedup got its own
regression test (the same gap reworded is suppressed, a different gap
chains). And the grader claimed its third natural phrasing false
negative, a correct "you will need to move off the basic tier" scored
as a miss. Substring needles are now demonstrably the wrong tool for
yes or no verdict cases in both directions; sweeps get hand adjudicated
until the harness grows a judge mode for those, and the fixture file
keeps growing either way.

The cheapest counter to the conflation half, entity scoping, was built and
measured next, and it failed in a way that is more informative than a
success would have been. The idea: gate the semantic expansion so a
neighbour is kept only if it shares a content entity (a token that is not a
stopword, a unit word, or a number) with the fault topic, so that
"gigabytes" alone cannot bridge the drive fact and the storage tier fact.
A unit test settles the theory before any model runs: the drive fault
shares no content entity with the storage 140 fact, so scoping would
correctly drop it, but the engineers fault also shares no content entity
with the "platform team" fact, so scoping would drop that too. Both needed
facts are disjoint from their fault in the identical way. A lexical filter
that fixes the conflation necessarily breaks the reach; it cannot tell the
two apart because lexically they are the same situation.

Then the measurement said something the theory had not. Isolating the drive
case with only expansion toggled, expansion off answers correctly (620
against 500, will not fit) and expansion on pulls the storage 140 into a
620 minus 140 calculation and concludes it fits: expansion is the sole
cause of the bleed when the transcript is sparse. But in the fuller
transcript with three facts competing, the trace shows no page fault at
all, only a calculation over 620 and 140. The model never faulted, because
base retrieval on the original question had already placed the storage 140
in the working set next to the drive numbers, and three plausible gigabyte
figures are enough for it to pick a wrong pair and compute with confidence.
The bleed there is upstream of the entire fault, expansion, and scope
machinery, so scoping the expansion changes nothing: there is nothing to
scope.

Two things fall out of that. The bleed has a density dependent entry point,
expansion in the sparse case and base retrieval in the dense one, so a fix
that only touches fault time cannot cover it. And more fundamentally,
conflation is invisible to the fault protocol: the whole design rests on
the model knowing when it is missing something, and a wrong-but-plausible
number handed over by retrieval is exactly the case where it does not know.
This is the same shape as the silent staleness in the date case. The fix
these point to is not a token filter and not a wider net but an entity
scoped base retrieval where "platform team" and "engineers" are linked and
"storage tier" and "drive" are not, which is the semantic entity graph in
the issue tracker, and nothing cheaper reaches it.

So the minimal version of that graph got built and measured, and the result
is the most useful kind of negative. At ingest each message attaches to its
content entities (units and numbers dropped) and each entity is embedded
once; base retrieval resolves a question's entities to the nearest entity
nodes and loads the messages hanging off them, replacing the lexical route
rather than adding to it. Against the lexical base on the eight disjoint
cases, with fault expansion off so the only variable is the base index, the
graph went four of eight against five, one worse. But the two cases that
moved tell the whole story. The drive question, which lexical retrieval got
wrong by pulling the storage tier's 140 into a 620 minus 140, the graph got
right: it loaded fifteen messages instead of thirty, the 140 was not among
them, and it answered 620 against 500, will not fit. The precision half of
the thesis is real and the mechanism is exactly the predicted one: the 140
message's only content entity is the vague "keeping", so it hangs off an
orphan node a drive query never reaches.

> **RETRACTED (this claim, right here).** "The precision half of the thesis
> is real / the mechanism is exactly the predicted one" did not survive
> re-running. On repeat runs the drive verdict flipped on identical
> retrieval, and the graph that had "fixed" drive bled the 140 the next
> time. The confident precision claim was a single lucky run under a
> noise-dominated judge. The correction and the deterministic replacement
> are two paragraphs below; this marker is here so the original claim is not
> read as standing.

The same orphaning is why the graph lost. The 140 that is a distractor to
the drive question is the answer to the storage tier question, and being
orphaned from everything it is now unreachable by the one question that
needs it, so storage tier went from pass to fail. Reach failed the same way
for the cases that need a cross vocabulary hop: "battery" did not resolve to
the laptop's charge fact and "engineering" did not resolve to the platform
team, because the only link on offer was raw embedding similarity between a
query word and an entity string, which is too weak across vocabulary and too
blunt across unit words, the identical failure in a new place.

The verdict is precise rather than a shrug. The cheap graph does not earn the
full build, it trades precision for reach and nets slightly worse. But it is
not a null result: it confirms the precision mechanism works and it localizes
the two things the shortcut left out, both load bearing. Extraction has to be
coreference aware, so "140 gigabytes up there" attaches to storage rather than
to "keeping". And entities need edges to each other, learned from
co-occurrence, so "engineers" and "platform team" become one reachable thing
without either sharing a word, which is the association graph and its cold
start problem exactly as the prior work warned. The next measurable step is
those edges, on this same harness. That is what decides the full build, and
the shortcut around it is now closed off with evidence rather than by
assertion. One run per condition here; the drive result is a single run, but
the mechanism is legible in the loaded counts rather than inferred from the
score.

That last hedge turned out to matter, and the paragraph above needs a
correction that is worth keeping visible rather than editing away. Re-running
the conditions, the verdicts flipped: the drive case passed under the lexical
baseline in one run and failed in the next on identical retrieval, and the
graph that had "fixed" drive now bled the 140. The judge-graded end-to-end was
noise dominated, the model and the grader together vary more than the effect
being measured, and the confident precision claim was a single lucky run. So
the model and the judge got cut out entirely and retrieval composition was
measured directly, which is deterministic. That probe is the real result.

It says two things, both firmer than anything the end-to-end runs supported.
The lexical route loads all fourteen planted facts on every query, because the
corpus is small enough that it never has to select, so the conflation is not
retrieval pulling in a wrong number, it is the model handed every number and
picking the wrong pair about half the time. And entity routing reaches only
one of the two facts each synthesis question needs, drive's 500 but not its
620, the engineers' 12 but not the platform team's 15, so its apparent
precision is empty because it drops the fact the question needs along with the
distractor. The exact shape of the failure: for the drive question the storage
tier's 140 is the third nearest entity in the whole store while the drive's
own 620 fact sits past rank thirty-five, so the wrong number is ten times
nearer than the right one and no threshold separates them. For disjoint
synthesis, embedding proximity is anti-correlated with relevance, and the cheap
graph does not earn the build, not for want of tuning but because the premise
that nearness tracks relevance is false here. Two process lessons fall out:
n=1 with a generative model and an LLM judge cannot resolve effects this size,
and a harness small enough that lexical loads everything cannot test precision
at all. The deterministic probe is the instrument that should have been built
first.

> **RETRACTED (the sentence just above).** "For disjoint synthesis, embedding
> proximity is anti-correlated with relevance" was stated as a general result.
> It is not one. It was an artifact of embedding entity *tokens* rather than
> passages. The next section shows the same nomic model, on the same corpus,
> ranking the drive's own 620 fact at rank 2 instead of past rank 35, purely
> by embedding the full passage instead of the token "library". So the
> anti-correlation was a property of our entity-token index, not of dense
> retrieval on this task. The finding it was used to support (that the cheap
> entity graph does not earn the build) still holds, but for the corrected
> reason below.

### Off-the-shelf graph RAG, and the attribution that killed the recommendation

The obvious next move was to stop hand-rolling a graph and evaluate a real one.
Two got stood up against the deterministic probe, with the model and judge kept
out: **HippoRAG 2.0.0a3** (its published code: OpenIE over local Ollama
qwen2.5:14b, native facebook/contriever embeddings, Personalized PageRank) and
**LightRAG 1.5.4** (LLM entity/relation extraction, dual-level retrieval).

First a correction to a claim made in ruling HippoRAG out: I had said it would
not install because torch has no Python 3.14 wheels. That was wrong. torch ships
3.14 wheels. HippoRAG's actual blocker is a hard pin to `torch==2.5.1`, which
resolves cleanly on Python 3.12, where it installs and runs. A second correction
in the same vein, because it was also overstated: on 3.12 the install "succeeded"
but `vllm==0.6.6.post1` has no macOS wheel at all. pip downloaded the sdist and
built a `py3-none-any` stub of 1.7 MB, a pure-Python shell with none of vllm's
compiled kernels. It is non-functional; it only did not matter because HippoRAG
ran inference through Ollama and never invoked local vllm. "Installed cleanly"
was the wrong phrase for a source build of a stub.

The result, ranking the second operand each synthesis question needs (the one
our entity graph missed) and the same-unit distractor:

| case (2nd operand) | our entity graph (nomic on entity tokens) | nomic PASSAGE dense | contriever PASSAGE dense | HippoRAG (contriever + PPR) |
|---|---|---|---|---|
| drive.b (620) | ~rank 40 | 2 | 2 | 1 |
| eng.b (platform team) | unreachable | 2 | 2 | 5 (graph hurt) |
| storage.b (140) | missed | 1 | 4 | 2 |
| laptop.b (6h) | 1 | 1 | 1 | 1 |

Plain passage-level dense retrieval, with the nomic model already in the system,
puts both operands into the top three on all four cases and matches or beats
HippoRAG. The attribution controls say why, and they are the point. HippoRAG's
own OpenIE triples show the two drive facts share no graph node (`external
drive/500` versus `photo library/620`), so PPR cannot bridge them; the good
drive rank is dense-driven, not traversal-driven. Pure contriever dense with no
graph ranks the operands the same or better than full HippoRAG, and on the
engineers case the graph made it worse, rank 2 to rank 5. And nomic at the
passage level equals contriever, so there is not even an embedding-model case
for switching. LightRAG, on 24 documents, returned the whole corpus for every
query, load-everything, indistinguishable from the lexical baseline.

So the entity graph in the sections above failed for one concrete reason: it
embedded entity tokens, which strips the context dense retrieval needs. The fix
that recovers three of the four failures is passage-level dense retrieval, which
is not a new capability and not a reason to integrate HippoRAG (whose local
OpenIE also hit a 21-of-24 parse-failure rate against qwen2.5:14b, because its
extractor is tuned to GPT-4o's output format).

### The passage-dense recommendation is already falsified

That last conclusion pointed at "switch to passage-level dense retrieval." A
reading of `hierarchical.rs` retires it before any code is written. `route_query`
already computes `s += cos(query_embedding, msg.embedding)` over the *full
message embedding*: passage-level dense scoring is already present. It is gated,
though, behind BM25 plus tree-beam candidate generation; the cosine only reranks
what those two already surfaced, and a passage neither surfaces is never
dense-scored. To reproduce the ungated probe that ranked drive.b at 2, the gate
would have to be removed and every passage dense-scored. Removing that gate is
exactly the change already built and measured under the name expansion
(`fault_semantic_expansion`), which bypasses the candidate gate with pure dense
neighbours and was measured pulling the storage 140 into the drive query and
worsening the conflation. Ungated passage dense does surface the missing operand
(good for reach) and the distractor sits between the two operands at rank 1
(feeds conflation) in the same motion. So the recommendation is not a live new
capability; it is a known-harmful lever the gate exists to prevent, and the
residual conflation stays a model-side problem for an answer-time guard, not a
retrieval one.

![recall after total eviction, exact answers at ~40ms](shots/stress-recall.png)

Notes from living with it: write back runs one extra model call per turn,
so replies take a few seconds longer than plain chat. The 8B model
sometimes decorates recalled facts; in one test it added a year to a date I
never gave it. The 8B write back classifier also files things under odd
topic names, which is why the store deduplicates on write and the browser
has correct and delete. A bigger classifier, or handing classification to
the answer model, fixes more of that than prompt tweaks do.
