import sys
from pathlib import Path
import pytest
import pandas as pd
import json
from networkx.readwrite import json_graph
import logging
import csv
import atexit
import traceback

current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from MythExtraction.MythExtractUtil import extract_knowledge_graph
from MythIsomorphism.MythIsomorphismUtil import extract_anonymized_triples, extract_surface_triples, get_similarity_scores

logger = logging.getLogger("test_logger")
logger.setLevel(logging.INFO)

DB_PATH = Path("D:/MythoGraph/MythoGraph/MythoGraph/MythoGraphDB")
CSV_PATH = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/fables_test_dataset.csv")
SCORES_CSV_PATH = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/similarity_scores_output.csv")

all_scores = []

@pytest.mark.parametrize("graph_id,title,content", [
    (row["Id"], row["Title"], row["Content"]) for _, row in pd.read_csv(CSV_PATH, encoding='ISO-8859-1').iterrows()
])
def test_extract_knowledge_graph_similarity(graph_id, title, content):
    predicted_graph = extract_knowledge_graph(content)
    predicted_graph_json = json_graph.node_link_data(predicted_graph)

    current_anonymized = extract_anonymized_triples(predicted_graph_json)
    current_surface = extract_surface_triples(predicted_graph_json)

    reference_filename = f"{graph_id}.json"
    reference_path = DB_PATH / reference_filename

    if not reference_path.exists():
        all_scores.append({
            "ID": graph_id,
            "Title": title,
            "AnonSim": "N/A",
            "SurfaceSim": "N/A",
            "JaccardSim": "N/A",
            "FinalScore": "N/A",
            "Result": "SKIPPED"
        })
        pytest.skip(f"[SKIP] Reference file not found: {reference_filename}")

    try:
        with open(reference_path, encoding="utf-8") as f:
            reference_graph = json.load(f)

        anon_sim, surf_sim, jaccard_sim = get_similarity_scores(reference_graph, current_anonymized, current_surface)
        final_score = round(0.7 * anon_sim + 0.2 * surf_sim + jaccard_sim * 10, 2)

        print(f"Score for {graph_id}: Anon={anon_sim:.2f}, Surface={surf_sim:.2f}, Jaccard={jaccard_sim:.4f}, Final={final_score:.2f}")

        all_scores.append({
            "ID": graph_id,
            "Title": title,
            "AnonSim": f"{anon_sim:.2f}",
            "SurfaceSim": f"{surf_sim:.2f}",
            "JaccardSim": f"{jaccard_sim:.4f}",
            "FinalScore": f"{final_score:.2f}",
            "Result": "PASS" if final_score >= 60.0 else "FAIL"
        })

        if final_score < 60.0:
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
            "JaccardSim": "ERROR",
            "FinalScore": "ERROR",
            "Result": f"ERROR: {str(e)}"
        })
        pytest.fail(f"[ERROR] Failed to compare {graph_id}: {str(e)}")

@atexit.register
def write_results_to_csv():
    if all_scores:
        with open(SCORES_CSV_PATH, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Title", "AnonSim", "SurfaceSim", "JaccardSim", "FinalScore", "Result"])  # Add JaccardSim here
            writer.writeheader()
            writer.writerows(all_scores)
        print(f"\nSimilarity scores written to: {SCORES_CSV_PATH}")