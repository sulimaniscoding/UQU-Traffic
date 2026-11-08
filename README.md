# UQU Traffic AI — Assignment Code Review

**Academic submission · selected source code only**  
**Project:** Predictive Traffic Management for Umm Al-Qura University  
**Status:** Research prototype using synthetic student data and SUMO traffic simulation

This repository is a **limited code-review extract** prepared for the course instructor. It shows the implemented data generation, geographic road routing, SUMO traffic simulation, machine-learning forecasting, and gate routing logic. It intentionally excludes the complete research workspace, raw student timetable datasets, trained model binaries, video recordings, SUMO road-network assets, and full run logs.

> **Important:** The code in `src/` is copied **unchanged** from the original project packages. Because the large input datasets, geographic network and model artifacts are intentionally excluded, this repository is **not a standalone runnable version** of the full experiment. It is for source inspection and evaluation. The complete project is maintained privately by the student.

## What to inspect

| Source file | Relevant implementation |
|---|---|
| [`src/generate_dataset.py`](src/generate_dataset.py) | Generation of fictional academic records and student schedules |
| [`src/network_setup.py`](src/network_setup.py) | OSM/SUMO network processing and shortest-path routing (Dijkstra-style) |
| [`src/experiment.py`](src/experiment.py) | Synthetic travel demand, SUMO simulation, Random Forest/XGBoost training, and gate policy comparison |
| [`src/report_and_validate.py`](src/report_and_validate.py) | Data integrity checks, evaluation, and experiment reporting |

### Key functions in `src/experiment.py`

- `prepare()` — normalize and derive academic demand inputs.
- `demand()` — turn synthetic student-day schedules into simulated vehicle trips.
- `simulate()` — collect traffic state and evaluate fixed, reactive, or predictive routing.
- `train()` — train and validate Random Forest and XGBoost using a **15-minute-ahead** target.
- `main()` — experiment command-line entry point.

The prediction target is the **number of stopped vehicles near each gate 15 minutes in the future**, not a real-world measured queue. The decision controller is **rule-based** and separate from the ML model. A separate YOLO/ByteTrack video prototype exists in the original research workspace but is **not part of this selected submission** and was not integrated into this simulation pipeline.

## Results included

- [`results/model_metrics.csv`](results/model_metrics.csv) — validation and test MAE/RMSE for Persistence, Random Forest and XGBoost.
- [`results/control_comparison.csv`](results/control_comparison.csv) — simulation policy comparison for normal, peak and incident scenarios.
- [`results/model_comparison.png`](results/model_comparison.png) and [`results/control_comparison.png`](results/control_comparison.png) — figures generated in the original experiment.

On the reported test split, **Persistence had the lowest MAE (1.541)**, while **Random Forest had the lowest RMSE (6.550)**. Random Forest was selected among the two ML candidates by validation MAE. Reactive routing performed better in the normal and peak policy comparisons; predictive routing performed best in the single reported incident case. These are exploratory **simulated** results and should not be generalized to actual campus traffic without field validation and repeated trials.

## Limitations

1. All academic student records and travel assumptions are **synthetic**. They are not official UQU student records.
2. The road network and gate configurations are modeled assumptions, not a deployed traffic control system.
3. Raw inputs, trained weights, replay recordings and other research materials are excluded deliberately.
4. No live cameras, barriers or campus gates are controlled by this repository.
5. Some source files import assets absent from this extract; inspect the source rather than expecting to execute the full workflow here.

See [`docs/CODE_WALKTHROUGH_AR.md`](docs/CODE_WALKTHROUGH_AR.md) for a short Arabic walkthrough and [`ORIGINAL_SOURCE_HASHES.txt`](ORIGINAL_SOURCE_HASHES.txt) for the source-file hashes.

**Access:** Keep this repository **Private** and invite the instructor as a collaborator if they need to inspect it. GitHub collaborator access is repository-wide; this submission is a separate repository so the original full project remains inaccessible.
