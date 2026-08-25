# Part I — Foundations

*Adversarial Robustness & Cross-Domain Transferability of Deep-Learning NIDS*

This part assumes no prior knowledge of either intrusion detection or adversarial machine
learning. Readers already familiar with both may skip to Part II (Methodology).

---

## 1. Network Intrusion Detection Systems

### 1.1 What a NIDS is

A **Network Intrusion Detection System** monitors traffic crossing a network and raises an
alarm when it observes something that looks like an attack. It sits at a choke point — a
span port on a core switch, a tap on the uplink, a virtual interface in a cloud VPC — and
reads packets that were going past anyway. It is a *detective* control: it reports, it does
not block. (A system that blocks is an Intrusion **Prevention** System; the detection
machinery is the same, the response differs.)

Two distinctions matter for everything that follows.

**Network-based vs host-based.** A NIDS sees traffic; a HIDS sees what happens *inside* a
machine — system calls, file changes, process trees. This work is entirely network-based.
The consequence is that the detector never observes the attacker's intent, only the traffic
their intent produced. That gap is what an adversary exploits.

**Packet-level vs flow-level.** A packet-level detector inspects individual packets,
including payload bytes. A flow-level detector aggregates packets sharing a **5-tuple**
(source IP, source port, destination IP, destination port, protocol) into a *flow*, and
works from summary statistics — how long the flow lasted, how many bytes travelled each
way, how many packets, at what rate.

All three datasets in this study are flow-level, and this is not an incidental choice.
Flow-level detection is what remains possible once traffic is encrypted: with TLS covering
most of the modern web, payload inspection is unavailable, and statistical properties of
the flow are all a detector has. It also scales — a 10 Gbps link carries far too many
packets to inspect individually, but far fewer flows.

> **Why this matters for adversarial ML.** A flow's feature vector is not a free-floating
> array of numbers. It is a *measurement* of something that physically happened on a wire.
> Section 4 returns to this at length, because it is the crux of Research Objective 2.

### 1.2 Signature-based detection

The traditional approach, and still the backbone of most production deployments (Snort,
Suricata, Zeek). An analyst writes a **signature** — a pattern describing known-malicious
traffic — and the detector matches every packet or flow against a rule database.

```
alert tcp any any -> $HOME_NET 445 (
    msg:"ETERNALBLUE SMB exploit attempt";
    content:"|FF|SMB|33 00 00 00 00 18 07 C0|";
    sid:2024218;
)
```

**Strengths.** Near-zero false positives on well-written rules. Fully explainable — the
alert names the rule that fired, so an analyst knows immediately what was detected and why.
Fast, and cheap to run.

**The fatal weakness.** A signature can only match an attack somebody has already seen,
analysed, and written a rule for. **Zero-day** attacks — those exploiting a previously
unknown vulnerability — pass straight through. So does any known attack that has been
trivially modified: change a byte in the payload, re-order the packets, add padding, and
the pattern no longer matches. This brittleness is what motivated statistical detection.

### 1.3 Anomaly-based and machine-learning detection

Instead of describing what is *bad*, learn what is *normal* and alarm on deviation. This
covers a spectrum:

* **Statistical anomaly detection.** Model the distribution of normal traffic; flag points
  in the tail. Simple, unsupervised, and prone to alarming on anything unusual-but-benign
  (a backup job, a software rollout, a new employee).
* **Classical machine learning.** Train a supervised classifier — decision trees, random
  forests, SVMs — on labelled benign and attack traffic. Requires feature engineering by a
  domain expert, but generalises to attack *variants* far better than signatures.
* **Deep learning.** Let the network learn its own feature representation. This is the
  subject of this study.

**Why deep learning for NIDS.** Attack traffic is characterised by high-order interactions
between features: it is not "long duration" or "many connections" that indicates a port
scan, but a specific *combination* of short duration, many distinct destination ports, few
bytes each, and a high SYN-error rate. Hand-engineering such conjunctions is laborious;
neural networks discover them.

**The catch — and it is a serious one.** Deep models are opaque, data-hungry, and, as this
study measures, adversarially fragile. A signature that fires is self-explanatory. A neural
network that fires produces a number, and an analyst must decide whether to trust it.

### 1.4 The base-rate problem

The single most important operational fact about intrusion detection, and the reason
accuracy is close to a meaningless metric here.

Consider a realistic link: 1,000,000 flows per day, of which 100 are malicious — a base
rate of 0.01%. Suppose a detector achieves a 99% detection rate at a 1% false-positive
rate. Those sound like excellent numbers.

|  | Actually malicious | Actually benign | Total |
|---|---:|---:|---:|
| **Alarm raised** | 99 | 9,999 | 10,098 |
| **No alarm** | 1 | 989,901 | 989,902 |
| **Total** | 100 | 1,000,000 | 1,000,100 |

Of 10,098 alarms, 99 are real. **Precision is 0.98%** — an analyst investigating alerts
from this detector wastes 99 investigations out of every 100. In practice the team stops
looking, and the detector might as well be switched off. This is *alert fatigue*, and it
kills more real deployments than poor detection rates do.

Two consequences run through this entire report:

1. **False-positive rate is reported everywhere**, and a low FPR is treated as more
   valuable than a marginally higher detection rate.
2. **Accuracy is never reported alone.** On a dataset that is 80% benign, a model that
   never alarms scores 80% accuracy while detecting nothing.

---

## 2. Adversarial Machine Learning

### 2.1 The core idea

Machine-learning models can be made to fail by inputs that a human would classify
correctly. The canonical demonstration (Szegedy et al., 2013; Goodfellow et al., 2015)
adds imperceptible noise to a panda photograph and the classifier confidently reports
"gibbon".

The mechanism is not mysterious. A trained classifier partitions its input space with a
**decision boundary**. In high dimensions, almost every point sits close to that boundary
along *some* direction — and the gradient of the loss points directly at the nearest one.
A small step in that direction crosses it.

Formally, for a model *f*, input *x*, true label *y*, and loss *L*, the adversary seeks:

> find **x'** minimising ‖x' − x‖ subject to *f*(x') ≠ *y*

or, in the budgeted form used throughout this study:

> find **x'** maximising *L*(*f*(x'), *y*) subject to ‖x' − x‖<sub>∞</sub> ≤ ε

The constraint ‖x' − x‖<sub>∞</sub> ≤ ε (the **L-infinity norm**) says: no single feature
may change by more than ε. Other norms appear in the literature — L2 bounds total
displacement, L0 bounds *how many* features change — but L∞ is standard for robustness
benchmarking and is what is used here.

### 2.2 Evasion vs poisoning

The two attack families differ in *when* the adversary acts, and the distinction structures
this entire study.

|  | **Evasion** | **Poisoning** |
|---|---|---|
| When | Test time (deployment) | Training time |
| What is attacked | The model's *input* | The model's *training data* |
| Model changed? | No | Yes — permanently |
| Analogy | Disguising yourself to pass a guard | Bribing the academy that trains the guards |
| Persistence | One flow at a time | Every future decision |
| NIDS example | Pad a port scan so it looks like a file transfer | Mislabel attack traffic as benign in the retraining set |

**Evasion** is the more studied. The adversary already has malicious traffic to send and
wants it classified benign.

**Poisoning** is arguably the more realistic threat for NIDS specifically, and is
under-studied. Intrusion detectors are *routinely retrained* on traffic harvested from the
network they protect, because traffic drifts and a stale model degrades. Labels for that
retraining come from imperfect sources: analyst triage, a signature engine's verdicts, or
the assumption that "traffic from last Tuesday was clean". An adversary with any influence
over that pipeline — including simply *being present on the network during the collection
window* — shapes the model itself.

### 2.3 Threat models: what the adversary knows

Attack strength depends entirely on assumed knowledge, and papers that blur this produce
incomparable numbers.

**White-box.** The adversary has the model: architecture, weights, gradients. This is the
strongest assumption and the least realistic, but it is the right one for *measuring
robustness* — it gives an upper bound on vulnerability. If a model survives a white-box
attack, it survives weaker ones. FGSM and PGD are white-box.

**Black-box.** The adversary sees only inputs and outputs, or nothing at all. This is the
realistic setting for a deployed NIDS: the model sits inside a vendor appliance or a
defender's infrastructure. The practical black-box technique is **transfer**: the adversary
trains their own *surrogate* model on similar public data, attacks that, and relies on
adversarial examples transferring between models that learned similar boundaries. Transfer
success rates are the headline operational number.

**Grey-box.** Something in between — architecture known but not weights, or query access
with a rate limit. Not separately studied here.

### 2.4 Why adversarial examples transfer

Transferability is initially surprising: why should an example crafted against model A fool
an unrelated model B?

The leading explanation (Ilyas et al., 2019) is that adversarial examples exploit
**non-robust features** — patterns that are genuinely predictive of the label in the
training distribution, but brittle. Any model trained on the same data learns the same
predictive patterns, including the brittle ones, so an attack that breaks one breaks
others. Transferability is not a quirk of a particular model; it is a property of the
*dataset*.

This has a direct consequence for RO1: if transfer is high across architectures, then
"deploy a different architecture" is not a defence, and architectural diversity buys
nothing. Phase 3 of the experiments measures exactly this.

---

## 3. Datasets

### 3.1 NSL-KDD

**Origin.** A cleaned revision (Tavallaee et al., 2009) of KDD Cup '99, itself derived from
a 1998 DARPA evaluation. Two files: `KDDTrain+` (125,973 records) and `KDDTest+` (22,544).

**Features.** 41 per connection, in three groups:
* *Basic* — duration, protocol, service, flag, bytes each way.
* *Content* — payload-derived counters (failed logins, root shell obtained). A legacy of
  the unencrypted 1990s; unobtainable on modern encrypted traffic.
* *Traffic* — windowed statistics over the preceding 2 seconds and the preceding 100
  connections (connection counts, error rates, same-service rates).

**Attack classes.** DoS, Probe (reconnaissance), R2L (remote-to-local), U2R
(user-to-root/privilege escalation). Binarised to benign/attack throughout this study.

**The property that makes it valuable — and the one most papers destroy.** `KDDTest+`
deliberately contains attack types **absent from** `KDDTrain+`. It is a built-in zero-day
generalisation test. The measured consequence, from this study's own control experiment:

| Evaluation | Logistic regression accuracy | ROC-AUC |
|---|---:|---:|
| Validation split (same distribution as training) | **97.98%** | 0.998 |
| Official `KDDTest+` (contains novel attacks) | **74.37%** | 0.776 |

A 23-point gap, from the same model on the same features. Any paper reporting ~99% on
NSL-KDD has shuffled the two files together and is measuring interpolation, not detection.
**This study preserves the official split.**

**Criticisms.** The traffic is synthetic and 25 years old; the content features are
unusable on encrypted traffic; the attack mix does not reflect a modern threat landscape.
It is retained here because it is small enough for exhaustive experimentation, universally
used (so results are comparable), and because its train/test gap is a genuine asset.

### 3.2 UNSW-NB15

**Origin.** Generated at UNSW Canberra in 2015 using the IXIA PerfectStorm traffic
generator — real modern background traffic with synthetic attacks injected. 2.5M records
total; the commonly used partitioned release has 175,341 training and 82,332 testing rows.

**Features.** 49, all flow-statistical: duration, byte and packet counts each direction,
TTL values, TCP window sizes, inter-packet timings, plus connection-state counters over
recent windows.

**Attack categories.** Nine — Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic,
Reconnaissance, Shellcode, Worms.

**Why included.** Modern, realistic, and — crucially for RO1 — generated by a *completely
different pipeline* from NSL-KDD. Cross-domain transfer between them is a genuine test of
whether learned representations survive a change of capture environment.

**Access.** Registration form; cannot be scripted.

### 3.3 CICIDS2017

**Origin.** Canadian Institute for Cybersecurity, 2017. Five days of captured traffic on a
purpose-built testbed, with benign behaviour generated from profiles of 25 real users.
~2.8M flows.

**Features.** 78, extracted by CICFlowMeter: forward/backward packet counts and lengths,
inter-arrival-time statistics (mean, std, min, max), flag counts, active/idle periods.

**Attack categories.** 14 — Brute Force, Heartbleed, Botnet, DoS variants, DDoS, Web
attacks, Infiltration, PortScan.

**Known data-quality defects**, corrected in `nids/data/loaders.py`:
* leading spaces in nearly every column name;
* infinite values in `Flow Bytes/s` and `Flow Packets/s` where duration is zero;
* NaN rows;
* severe class imbalance (~80% benign) — hence the base-rate discussion in §1.4.

**Why included.** The most modern and realistic of the three, and the largest.

### 3.4 The cross-domain problem

The three datasets do not share a feature space. This study computes the *genuine*
intersection (`nids/data/schema.py::shared_features`) rather than assuming one:

| Dataset pair | Shared canonical features |
|---|---:|
| NSL-KDD ∩ UNSW-NB15 | 10 |
| NSL-KDD ∩ CICIDS2017 | 5 |
| UNSW-NB15 ∩ CICIDS2017 | 7 |
| **All three** | **5** |

The three-way intersection is just `duration`, `protocol_type`, `src_bytes`, `dst_bytes`,
`dst_host_count`.

This is a finding, not an obstacle to be engineered around. Two common workarounds both
invalidate the result and are **deliberately not used**:

* **Zero-filling absent features.** The model learns "feature 47 is always zero, therefore
  I am in domain A" — a domain detector, not a transfer result.
* **Fine-tuning on target data.** That measures domain *adaptation*, a different and much
  easier question.

Cross-domain numbers in Part IV are therefore low. That low number *is* the contribution:
it quantifies how poorly public-dataset NIDS results generalise to a different network,
which is precisely the question facing anyone who wants to deploy one.

---

## 4. Realistic Constraints: Why Network Traffic Is Not an Image

This section motivates Research Objective 2 and is the conceptual heart of the study.

### 4.1 The false analogy

Adversarial ML was built on images, where the threat model is simple and *correct*: any
pixel may take any value in [0, 255], every combination is a displayable image, and a small
L∞ perturbation is invisible to a human. The domain is a **box** — a product of independent
intervals — and any point inside it is realizable.

Network flow features are not a box. Applying an unmodified gradient attack to them
produces records like these, all observed in this study's own unconstrained runs:

| Proposed adversarial value | Why it cannot exist |
|---|---|
| `duration = −0.4 s` | Time does not run backwards |
| `src_bytes = 1204.7` | Bytes are integers |
| `protocol_type = 0.3·tcp + 0.7·udp` | A packet has exactly one protocol |
| `src_bytes = 12,000,000` with `duration = 0.001` | ≈ 12 GB in one millisecond |
| `serror_rate = 0.83` with `count = 1` | A rate over one connection is 0 or 1 |
| `dst_bytes` reduced by the attacker | The *victim* sent those bytes |

An adversarial example that cannot be transmitted is not an attack. A detector's "failure"
on such a point is a mathematical artefact. **Papers that omit these constraints
systematically overstate attack success rates**, and Part IV quantifies by how much.

### 4.2 Six constraint families

Implemented in `nids/attacks/constraints.py` as a projection operator Π mapping any
perturbed point to the nearest realizable one. Applied after every gradient step, so the
iterate is feasible throughout — the standard projected-gradient formulation, with the
feasible set defined by network semantics rather than by an ε-ball alone.

**1. Immutability.** Some features are not the attacker's to change:
* *Attack-defining* — `protocol_type`, `service`. A TCP SYN flood switched to UDP is a
  different attack, not an evasion of the same one.
* *Victim-generated* — `dst_bytes`, `dst_packets`. The target host produced these.
* *Network-observed* — `flag` (connection state) reflects how the handshake actually went.

**2. Directionality — the most consequential constraint.** Real attackers can **add** but
not **remove**. They can pad payloads, inject packets, insert delays, open extra
connections. They cannot un-send a packet or retroactively shorten a duration. Features
such as `duration`, `src_bytes`, `count` are therefore **increase-only**: perturbations may
raise them but never lower them below the original value.

This single constraint removes roughly half the gradient directions available to the
attacker, and Part IV shows it is responsible for most of the drop in attack success.

**3. Box bounds.** Every feature clipped to its valid range — [0, 1] after this study's
scaling, non-negative in physical units.

**4. One-hot integrity.** A categorical feature encoded across several columns must have
exactly one active. Gradient steps produce fractional mixtures; the projection restores a
hard argmax.

**5. Integrality.** Counts are integers. Critically, rounding happens in **original units**,
not scaled ones: under the log1p scaling used here, one scaled unit spans a fraction of a
byte near zero and millions of bytes near one, so rounding the scaled value is meaningless.

**6. Dependency repair.** Derived features must stay arithmetically consistent with their
parents. A rate computed over *n* connections must be a multiple of 1/*n* — `serror_rate =
0.37` with `count = 2` is impossible and is snapped to 0.5. This is a *sound* repair (it
never accepts a record it should reject) though weaker than full generative reconstruction,
since the underlying per-connection outcomes are not recoverable from a single record.

### 4.3 The scaling decision

Not cosmetic, and worth stating explicitly.

Flow features are extremely heavy-tailed: `src_bytes` in NSL-KDD spans 0 to ~1.4 billion
with a **median of 44**. Under standardisation (z-score), almost all mass compresses into a
sliver near zero while a handful of outliers dominate the range — and a fixed ε then means
wildly different things for different features.

This study uses **log1p followed by min-max to [0, 1]**, for three reasons:

1. it compresses the heavy tail into a usable range;
2. a perturbation budget in scaled space becomes a *multiplicative* change in bytes — which
   is the physically meaningful notion for traffic volume (adding 100 bytes to a 44-byte
   flow is a large change; to a 1 MB flow it is nothing);
3. every feature shares [0, 1], so a single L∞ budget is comparable across features.

Because log1p is monotone, order is preserved — so "no smaller than the original" is the
same test in scaled space as in bytes, which is what makes the directionality constraint
implementable as a simple element-wise maximum.

---

## 5. Glossary

**Adversarial example** — an input deliberately perturbed to cause misclassification, while
remaining (by some measure) close to a correctly-classified input.

**Alert fatigue** — analyst desensitisation caused by excessive false positives; the usual
practical cause of a NIDS being ignored or disabled.

**Attack success rate (ASR)** — fraction of *correctly-classified* attack samples that an
attack flips to benign. Computed only over samples the model originally got right, since
flipping an already-missed sample is no achievement.

**Base rate** — the true prevalence of attacks in traffic. Typically <0.1%, which is why
precision collapses even at low false-positive rates (§1.4).

**Black-box attack** — adversary has no access to model internals; typically implemented by
attacking a surrogate and transferring.

**Clean-label poisoning** — training-time attack that perturbs *features* while leaving
labels correct, so every poisoned record survives human review.

**Detection rate (recall, TPR)** — fraction of real attacks the detector catches.

**Epsilon (ε)** — the perturbation budget; the maximum permitted change per feature under
L∞.

**Evasion attack** — test-time perturbation causing malicious input to be classified benign.

**False positive rate (FPR)** — fraction of benign flows wrongly alarmed. The metric that
determines deployability.

**FGSM** — Fast Gradient Sign Method; single-step white-box attack, `x + ε·sign(∇ₓL)`.

**Flow** — packets sharing a 5-tuple, summarised statistically. The unit of analysis here.

**Label flipping** — poisoning by changing labels on part of the training set.

**L∞ norm** — maximum absolute change across features; the standard adversarial budget.

**Non-robust features** — genuinely predictive but brittle patterns; the leading explanation
for why adversarial examples transfer between models.

**PGD** — Projected Gradient Descent; iterative FGSM with projection back into the ε-ball
each step. The standard strong white-box benchmark.

**Poisoning attack** — training-time corruption that permanently degrades the deployed model.

**Projection operator (Π)** — maps an arbitrary point to the nearest point in a feasible
set; here, the set of realizable network flows.

**ROC-AUC** — threshold-free ranking quality; probability a random attack is scored above a
random benign flow.

**Signature-based detection** — pattern-matching against known-attack rules; precise but
blind to novel attacks.

**Surrogate model** — attacker's own model, trained on similar data, used to craft
transferable adversarial examples.

**Transferability** — the degree to which adversarial examples crafted on one model fool a
different one.

**White-box attack** — adversary has full model access including gradients; gives an upper
bound on vulnerability.

**Zero-day** — an attack exploiting a previously unknown vulnerability, therefore having no
signature.
