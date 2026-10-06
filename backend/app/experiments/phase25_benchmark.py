"""Fresh, manually source-checked questions for the Week 02 slide corpus.

Page references are provenance, not guessed chunk IDs. Resolve them against
the actual chunks. Each cited slide is currently a single short chunk; refuse
automatic remapping if that assumption changes, rather than invent labels.
"""
from app.schemas.benchmark import BenchmarkItem
from app.generation.prompts import INSUFFICIENT_INFORMATION

# question, reference answer, supporting PDF pages, category, difficulty
QUESTIONS = [
    ("What doubling interval and cost trend does the slide give for transistor count under Moore's Law?",
     "Transistor count doubles roughly every two years with a minimal increase in cost.", [3], "factual", "easy"),
    ("In the RAM example, what does the 100 ns latency measure?",
     "The waiting time before the first byte arrives.", [9], "definition", "easy"),
    ("How do latency and bandwidth differ, and which direction of change is preferable for each?",
     "Latency is the wait before the first result and lower is better. Bandwidth is data transferred per unit time and higher is better.", [9,10], "comparison", "medium"),
    ("What annual transistor-density increase and approximate four-year change are stated in the slides?",
     "Density increases about 35% per year and roughly quadruples in four years.", [8], "factual", "easy"),
    ("Which memory technology is described as the foundation of main memory?",
     "DRAM is the foundation of main memory.", [11], "factual", "easy"),
    ("How is flash memory characterized in the slides, and is it volatile?",
     "Flash is electrically erasable programmable read-only memory and is nonvolatile.", [12], "definition", "easy"),
    ("According to the bandwidth/latency trend slide, how does bandwidth improvement compare with latency improvement?",
     "Bandwidth grows at least as the square of the improvement in latency.", [15], "comparison", "medium"),
    ("Does processor power stay constant across idle, normal, heavy, and burst workloads? Describe the stated pattern.",
     "No. Power is low when idle, moderate during normal work, high under heavy load, and very high during short bursts.", [17], "comparison", "easy"),
    ("What frequency and voltage does the example CPU use in normal and boost modes?",
     "Normal mode is 3.5 GHz at 1.0 V; boost mode is 5.0 GHz at 1.2 V.", [18], "factual", "easy"),
    ("What does TDP stand for, and what design purpose does it serve?",
     "Thermal Design Power is the sustained power target used to design the processor cooling system.", [19], "definition", "medium"),
    ("For the slide's 100 W TDP processor, what cooling capacity is suggested and what general rule is stated?",
     "Cooling should match or exceed TDP; the example suggests around 120 W or more depending on design.", [22], "factual", "medium"),
    ("Why can a higher-power processor still be more energy-efficient for a particular task?",
     "Energy, rather than power alone, is the better task-level comparison. A higher-power processor can be more energy-efficient if it completes the task significantly faster.", [23,24], "multi_hop", "medium"),
    ("For CMOS switching, how does transistor dynamic energy depend on load and voltage?",
     "It is proportional to capacitive load multiplied by voltage squared.", [25], "definition", "medium"),
    ("For a fixed task, contrast slowing the clock with lowering voltage in terms of dynamic power and energy.",
     "Slowing the clock reduces power but not energy for a fixed task. Lowering voltage reduces both dynamic power and energy.", [26], "comparison", "medium"),
    ("With unchanged capacitive load, what fraction of dynamic energy remains after a 15% voltage reduction?",
     "Dynamic energy scales with voltage squared, so 0.85 squared = 0.7225, or 72.25%, remains.", [25], "reasoning", "hard"),
    ("If both voltage and transition frequency drop by 15% with load unchanged, what fraction of dynamic power remains?",
     "Dynamic power scales with voltage squared times frequency: 0.85 squared times 0.85 = 0.614125, or about 61.41%, remains.", [25,26,27], "multi_hop", "hard"),
    ("What constraints do the slides give for slowing clock-frequency growth?",
     "Voltage cannot be reduced further and power per chip cannot increase because the air-cooling limit has been reached; heat dissipation is the major constraint.", [30], "reasoning", "medium"),
    ("How do clock gating and power gating differ in what they turn off?",
     "Clock gating stops the clock to inactive modules or cores; power gating turns off power to inactive modules.", [31,35], "comparison", "medium"),
    ("How does DVFS respond to a computationally intensive task compared with a less intensive task?",
     "It increases voltage and frequency for intensive work and adjusts them downward when high speed and throughput are unnecessary.", [32], "definition", "medium"),
    ("Why might a Turbo-enabled program's performance vary with room temperature, according to the slides?",
     "Turbo permits a higher clock rate only while thermally safe and until temperature rises, so room temperature affects the available boost and therefore performance.", [34], "reasoning", "hard"),
    ("What is the system-level rationale of race-to-halt?",
     "A faster, less energy-efficient processor can finish sooner and allow the rest of the system to halt.", [36], "definition", "medium"),
    ("Under an SLA, what distinguishes service accomplishment from service interruption?",
     "Service accomplishment delivers service as specified; service interruption delivers service different from the SLA.", [38], "comparison", "easy"),
    ("Which transition is a failure, which is a restoration, and how does MTTF relate to reliability?",
     "Failure moves from service accomplishment to interruption; restoration moves back. MTTF measures reliability through time to failure or continuous service accomplishment.", [38,39], "multi_hop", "hard"),
    ("If a component has MTTF of one million hours, what failure rate follows from the slides' reciprocal relationship, in failures per hour and FIT?",
     "The rate is 1/1,000,000 = 0.000001 failures per hour, equivalent to 1,000 failures per billion hours, or 1,000 FIT.", [43], "reasoning", "hard"),
    ("What exact dollar penalty per hour does the lecture's SLA example impose?", INSUFFICIENT_INFORMATION, [], "unanswerable", "easy"),
    ("What is the exact maximum junction temperature in degrees Celsius of the 3.3 GHz Core i7 discussed in the slides?", INSUFFICIENT_INFORMATION, [], "unanswerable", "medium"),
    ("Which semiconductor process-node size in nanometers was used for the example CPU's low-power, normal, and boost modes?", INSUFFICIENT_INFORMATION, [], "unanswerable", "medium"),
    ("What numerical L1 cache hit latency is specified for the example Core i7?", INSUFFICIENT_INFORMATION, [], "unanswerable", "easy"),
    ("What exact fan rotational speed is prescribed for the example 100 W TDP cooling system?", INSUFFICIENT_INFORMATION, [], "unanswerable", "medium"),
    ("What measured percentage energy saving did a controlled race-to-halt experiment achieve in this lecture?", INSUFFICIENT_INFORMATION, [], "unanswerable", "hard"),
]


def build_benchmark(chunks):
    by_page = {}
    for chunk in chunks:
        by_page.setdefault(chunk.page_number, []).append(chunk)
    items = []
    for index, (question, truth, pages, category, difficulty) in enumerate(QUESTIONS, 1):
        if any(len(by_page.get(page, [])) != 1 for page in pages):
            raise ValueError("Source evidence page missing or split: manually review relevance mapping")
        items.append(BenchmarkItem(
            question_id=f"ca_small_{index:03d}", question=question, ground_truth=truth,
            answerable=bool(pages), relevant_chunk_ids=[by_page[p][0].chunk_id for p in pages],
            category=category, difficulty=difficulty,
        ))
    validate_benchmark(items, chunks)
    return items


def validate_benchmark(items, chunks):
    ids = {chunk.chunk_id for chunk in chunks}
    if len(ids) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    if len({item.question_id for item in items}) != len(items):
        raise ValueError("Duplicate question IDs")
    if len({item.question.casefold() for item in items}) != len(items):
        raise ValueError("Duplicate questions")
    for item in items:
        if not item.ground_truth or not item.ground_truth.strip():
            raise ValueError("Every item requires ground truth")
        if item.answerable != bool(item.relevant_chunk_ids):
            raise ValueError("Relevance IDs must match answerability")
        if not set(item.relevant_chunk_ids) <= ids:
            raise ValueError("Unknown relevance ID")
