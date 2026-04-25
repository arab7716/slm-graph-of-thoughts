### Optimizing Graph of Thoughts Reasoning on Small Language Models via Proactive Failure Mitigation

**Author:** Artha Gunasekera Abeysinghe  

**Advisor:** [Vikram Ramaswamy](https://www.cs.princeton.edu/~vr23/)

**Submitted:** April 2026

Advanced LLM reasoning structures, specifically Graph of Thoughts (GoT), enhance Large Language Model capabilities by modeling complex problems as a directed graph. However, deploying GoT on open-weight Small Language Models (SLMs) in the 7B--14B parameter range on local GPUs introduces accuracy and speed bottlenecks. In particular, executing GoT locally results in graph nodes executing sequentially rather than in parallel, leading to unmitigated generation of redundant, erroneous reasoning branches that waste computation. Furthermore, SLMs tend towards "Diversity Collapse" during reasoning, repeatedly generating the same incorrect outputs, which corrupts the graph downstream.

To resolve these limitations, this thesis proposes a novel Proactive Mitigation Framework designed to optimize GoT execution on local SLMs. Specifically, we reengineer the architecture by introducing a mid-generation intervention pipeline that uses an $O(1)$ similarity heuristic to detect low thought diversity and halt the generation of redundant branches. After Diversity Collapse is detected, our pipeline routes the redundant thought to an LLM Judge for verification of output accuracy, followed by prompting through a set of zero-shot Mixture-of-Experts (MoE) personas. 

We evaluate the performance of our architecture by ablation of its components. Our results across GoT tasks (Sorting, Set Intersection, and Keyword Counting) highlight that: 

* The similarity heuristic reduces execution time by up to 50% without degrading task performance.
* The LLM Judge introduces a time trade-off and demonstrates high capability to identify correct outputs, and struggles to correctly identify incorrect outputs.
* MoE persona prompting increases thought diversity compared to a generic retry prompt and doubles node recovery rates across all tasks. Furthermore, it achieves noticeably higher overall performance on certain tasks, improving accuracy in Set Intersection by up to 15%.

Ultimately, this thesis provides an in-depth study of the reasoning capabilities and limitations of SLMs with the GoT architecture and the potential of our Proactive Mitigation Framework for more efficient computation and improved task performance. 

Full Paper can be found here: [LINK](https://drive.google.com/file/d/1aKBAYSCPloKjH9n9QgxEo16cnbtot6TQ/view?usp=sharing).
