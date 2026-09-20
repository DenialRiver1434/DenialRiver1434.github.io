## Mechanistic Interpretability Benchmark Close Reading Notes

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