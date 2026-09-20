# Replicating Results from Circuit Localization Track of MIB

Reference paper - [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/pdf/2504.13151)

## Overview

In order to prepare for benchmarking the computational graph condensation algorithm against MIB, I replicated the key results in the Circuit Localization Track from the MIB paper by recreating values noted under Table 2.

## Journal (w/ Problems Encountered)

Below is a journal of my progress. Problems encountered are **<u>bolded and underlined</u>**, and I have written up how I resolved them.

### MIB Close Reading

Below are the notes I took as I read. The original notes were taken on OneNote and I had ChatGPT rewrite it in mardown. 

<details>
<summary>
Close reading notes
</summary>


**Definition (MIB): Mechanistic Interpretability Benchmark**

Goal: Score mech. interp. methods by how precisely and concisely they recover relevant causal pathways.
- **Definition (causal): Stuff that actually matter cause AND effect**

Overall Structure
- Methods weighed against track 1 or 2 depending on what they do
- *Track 1: Circuit localization track for methods that*
  - Locate which parts of the neural network are responsible for what tasks
  - Look at how they connect to form a circuit
- *Track 2: Causal variable localization track for methods that*
  - Make hidden vectors into features
  - Looks at whether the features can be made task-relevant
  - Stuff like SAEs, DAS
- Four tasks
- Five models

Task Structure
- Various reasoning types, difficulty, 1output format
- Various familiarity
- They have training/validation
  - Users can discover circuits, casual variables on training sets; use validation for hyperparameters
- Public test set for quick training; upload to private testset API for testing

Four Tasks
1. IOI: Complete the object of a sentence
2. Arithmetic: +/− of all possible pairs $(x,y) \in \{0,\ldots,99\}^2$
3. MCQA: **Information to answer question is in the prompt**, test how the model does with MCQ in general
4. AI2 Reasoning Challenge (AIC): Test the model knowledge of science

**Methodology**
- Start by assessing how the model performs the task for a baseline
- Both tracks use **activation patching**
  - *Circuit localization track uses activation patching to test if messing with stuff outside circuit does not change prediction*
  - *Casual variable localization uses activation patching to see if messing with a variable has the intended effect*
  - The $f$: original input -> counterfactual input is fixed so the comparison is fair
- Use private test case to generate leaderboard

*Track 1: Circuit Localization Track specifics*
- *Circuit $C$ is a computational graph $N$ where*
  - *The nodes of $N$ are either an attention head (e.g. lay.59 head 23) or a layer of MLP*
  - *Edges are abstract, reflects how information flows*
  - **Definition (CPR): Find components that positively effect the behavior**
    - *Higher score is better*
    - *Find components encouraging behavior*
  - **Definition (CMD): Find components that impact the behavior at all**
    - *Lower score is better*
    - *Explain the full algorithm*
- *Measuring faithfulness: $f(C,N)=\frac{m(C)-m(\emptyset)}{m(N)-m(\emptyset)}$ where $m$ is the resulting logit difference $y'-y$ of correct/incorrect inputs*
- *To capture faithfulness and minimality at the same time*
  - *Look at (circuit size, faithfulness score) at different $\lambda$ thresholds*
  - *$k$ is the model size*
  - *$CPR=\int_0^1 f(C_k)\,dk$, **higher score better***
  - *$CMD=\int_0^1 |1-f(C_k)|\,dk$, **lower score better***
- *Size defined as $k=|C|=\sum_{(u,v)\in C}\left(\frac{|N_u\cap N_C|}{|N_u|}\right)$*
- *Three types of circuit discovery MIB was tested on*
  - *Attribution methods (EAP, EAP-IG, NAP, ...): Way of speeding up activation patching*
  - *Information Flow Routes (IFR): If the output of $u$ is similar to input of $v$, then let $(u,v)$ be in the circuit*
  - *Mask-based method (UGS): A model that tries to learn a pruning mask that cuts stuff from the curicuit*
- *Results*
  - *See table 2; EAP-IG achieves best score, IFR performs poorly*

(notes on track 2 ommitted because it is irrelevant to computational graph condensation)

Links
- Dataset https://huggingface.co/collections/mib-bench/mib-datasets
- Code https://github.com/aaronmueller/mib
- Leaderboard https://huggingface.co/spaces/mib-bench/leaderboard
</details>

### MIB Circuit Setup

I used forked the repository containing the source code for the circuit track [into here](https://github.com/DenialRiver1434/MIB-circuit-track) then cloned it onto my VSC environment.

I spent plenty of time trying to install the dependencies due to the following problem.

**<u>Problem: The readme.md from the original repository does not contain the full instructions for installing dependencies</u>**

When I tried going through the steps, pip install . installed the wrong eap package. After checking the folders, it had installed EAP 1.1.14.1, an unrelated library. I asked Codex how to solve this and it told me to run ```pip install ./EAP-IG .``` which worked better.

Overall I had extra trouble because my primary laptop was broken and was working from a backup computer with weird installation settings.

