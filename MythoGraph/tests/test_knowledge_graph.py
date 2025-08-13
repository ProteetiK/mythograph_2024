import sys
import re
import csv
import json
import atexit
import logging
import traceback
import pytest
import pandas as pd
from pathlib import Path
from networkx.readwrite import json_graph

current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from MythExtraction.MythExtractUtil import extract_knowledge_graph
from MythIsomorphism.MythIsomorphismUtil import get_similarity_scores, compute_text_similarity

DB_PATH = Path("D:/MythoGraph/MythoGraph/MythoGraph/MythoGraphDB")
CSV_PATH = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/fables_test_dataset.csv")
SCORES_CSV_PATH = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/similarity_scores_output.csv")



logger = logging.getLogger("test_logger")
logger.setLevel(logging.INFO)

all_scores = []
weight_anonymized = 0.6
weight_surface = 0.05
weight_opposition = 0.2
weight_text = 0.1
weight_jaccard = 0.05
similarity_threshold = 60.0

@pytest.mark.parametrize("graph_id,title,content", [
    (row["Id"], row["Title"], row["Content"])
    for _, row in pd.read_csv(CSV_PATH, encoding='ISO-8859-1').iterrows()
])
def test_extract_knowledge_graph_similarity(graph_id, title, content):
    predicted_graph = extract_knowledge_graph(content)
    predicted_graph_json = json_graph.node_link_data(predicted_graph)

    reference_filename_prefix = str(graph_id)
    matching_files = [
        f for f in DB_PATH.glob("*.json")
        if re.match(rf"^{re.escape(reference_filename_prefix)}(\s|$)", f.stem)
    ]

    if not matching_files:
        all_scores.append({
            "ID": graph_id,
            "Title": title,
            "AnonSim": "N/A",
            "SurfaceSim": "N/A",
            "OppositionSim": "N/A",
            "TextSim": "N/A",
            "JaccardSim": "N/A",
            "FinalScore": "N/A",
            "Result": "SKIPPED"
        })
        pytest.skip(f"[SKIP] No matching file found for ID prefix: {reference_filename_prefix}")

    reference_path = matching_files[0]

    try:
        with open(reference_path, encoding="utf-8") as f:
            reference_graph = json.load(f)

        current_text = reference_graph.get("myth_text", "")
        other_text = predicted_graph_json.get("myth_text", "")

        anon_sim, surf_sim, opposition_sim, jaccard_sim = get_similarity_scores(reference_graph, predicted_graph_json)
        text_sim = 1.0

        final_score = round(
            weight_anonymized * anon_sim +
            weight_surface * surf_sim +
            weight_opposition * opposition_sim +
            weight_text * text_sim +
            weight_jaccard * jaccard_sim * 100, 2
        )

        print(
            f"Score for {graph_id}: "
            f"Anon={anon_sim:.2f}, Surface={surf_sim:.2f}, "
            f"Opposition={opposition_sim:.2f}, Text={text_sim:.2f}, "
            f"Jaccard={jaccard_sim:.4f}, Final={final_score:.2f}"
        )

        all_scores.append({
            "ID": graph_id,
            "Title": title,
            "AnonSim": f"{anon_sim:.2f}",
            "SurfaceSim": f"{surf_sim:.2f}",
            "OppositionSim": f"{opposition_sim:.2f}",
            "TextSim": f"{text_sim:.2f}",
            "JaccardSim": f"{jaccard_sim:.4f}",
            "FinalScore": f"{final_score:.2f}",
            "Result": "PASS" if final_score >= similarity_threshold else "FAIL"
        })

        if final_score < similarity_threshold:
            pytest.fail(f"Similarity too low for {graph_id} (score={final_score:.2f})")

    except Exception as e:
        print("======== FULL TRACEBACK ========")
        traceback.print_exc()
        print("================================")
        all_scores.append({
            "ID": graph_id,
            "Title": title,
            "AnonSim": "ERROR",
            "SurfaceSim": "ERROR",
            "OppositionSim": "ERROR",
            "TextSim": "ERROR",
            "JaccardSim": "ERROR",
            "FinalScore": "ERROR",
            "Result": f"ERROR: {str(e)}"
        })
        pytest.fail(f"[ERROR] Failed to compare {graph_id}: {str(e)}")

@atexit.register
def write_results_to_csv():
    if all_scores:
        with open(SCORES_CSV_PATH, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ID", "Title", "AnonSim", "SurfaceSim", "OppositionSim", "TextSim", "JaccardSim", "FinalScore", "Result"
            ])
            writer.writeheader()
            writer.writerows(all_scores)
        print(f"\nSimilarity scores written to: {SCORES_CSV_PATH}")