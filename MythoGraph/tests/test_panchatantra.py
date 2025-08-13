import json
import sys
from pathlib import Path
from networkx.readwrite import json_graph

current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from MythIsomorphism.MythIsomorphismUtil import get_similarity_scores

# folder_path = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/panchatantra")
# file1 = folder_path / "Lion and Crane manual.json"
# file2 = folder_path / "The_Lion_and_the_Crane_knowledge_graph.json"

folder_path = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/Ge")
file1 = folder_path / "Ge_-_Variation_-_1_txt_knowledge_graph.json"
file2 = folder_path / "Ge_-_Variation_-_2_txt_knowledge_graph.json"

# folder_path = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/aesops")
# file1 = folder_path / "aesops2 Bat Weasel.json"
# file2 = folder_path / "The_Bat_And_The_Weasels_knowledge_graph.json"
# file1 = folder_path / "aesops18 Hare Tortoise.json"
# file2 = folder_path / "The_Hare_and_the_Tortoise_knowledge_graph.json"

# folder_path = Path("D:/MythoGraph/MythoGraph/MythoGraph/tests/shakuntala")
# file1 = folder_path / "Shakuntala_-_Kalidasa_knowledge_graph.json"
# file2 = folder_path / "Shakuntala_-_Mahabharata_knowledge_graph.json"

weight_anonymized = 0.6
weight_surface = 0.05
weight_opposition = 0.2
weight_text = 0.1
weight_jaccard = 0.05

with open(file1, encoding="utf-8") as f:
    graph1 = json.load(f)

with open(file2, encoding="utf-8") as f:
    graph2 = json.load(f)

anon_sim, surf_sim, opposition_sim, jaccard_sim = get_similarity_scores(graph1, graph2)

text_sim = 100.0

final_score = round(
    weight_anonymized * anon_sim +
    weight_surface * surf_sim +
    weight_opposition * opposition_sim +
    weight_text * text_sim +
    weight_jaccard * jaccard_sim * 100,
    2
)

print(f"AnonSim:      {anon_sim:.2f}")
print(f"SurfaceSim:   {surf_sim:.2f}")
print(f"Opposition:   {opposition_sim:.2f}")
print(f"TextSim:      {text_sim:.2f}")
print(f"JaccardSim:   {jaccard_sim:.4f}")
print(f"Final Score:  {final_score:.2f}")