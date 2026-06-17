import spacy
import networkx as nx
from pyvis.network import Network
import os

# Load NLP model once
nlp = spacy.load("en_core_web_sm")

def build_graph_from_text(text):
    """
    1. Reads the text.
    2. Finds characters (Entities).
    3. Connects them if they are in the same sentence.
    4. Returns a NetworkX graph.
    """
    doc = nlp(text)
    G = nx.Graph()
    
    # 1. Extract Entities (People, Groups, Places)
    # We filter for PERSON, ORG (Groups), and GPE (Places)
    valid_labels = ["PERSON", "ORG", "GPE"]
    
    for sent in doc.sents:
        # Find all entities in this specific sentence
        sent_ents = [ent.text for ent in sent.ents if ent.label_ in valid_labels]
        sent_ents = list(set(sent_ents)) # Deduplicate
        
        if len(sent_ents) > 1:
            # Connect every entity to every other entity in this sentence
            for i in range(len(sent_ents)):
                for j in range(i + 1, len(sent_ents)):
                    source = sent_ents[i]
                    target = sent_ents[j]
                    
                    # Add nodes
                    G.add_node(source, title=source, group=1, color="#FFC107") # Gold
                    G.add_node(target, title=target, group=1, color="#FFC107")
                    
                    # Add edge (or increase weight if already exists)
                    if G.has_edge(source, target):
                        G[source][target]['weight'] += 1
                    else:
                        G.add_edge(source, target, weight=1, color="#555555")
                        
    return G

def save_graph_html(G):
    """
    Saves the interactive graph to a local HTML file.
    """
    if len(G.nodes) == 0:
        return None

    # Create PyVis Network
    net = Network(height="350px", width="100%", bgcolor="#0E1117", font_color="white")
    net.from_nx(G)
    
    # Physics settings (makes it bounce nicely)
    net.repulsion(
        node_distance=120, 
        central_gravity=0.2, 
        spring_length=150, 
        spring_strength=0.05
    )
    
    # Save to src folder
    output_path = os.path.join("src", "graph.html")
    net.save_graph(output_path)
    return output_path