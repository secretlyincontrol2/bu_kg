from database import db

def discover():
    # 1. Labels and their properties
    print("--- Labels and Properties ---")
    query_labels = "MATCH (n) RETURN DISTINCT labels(n) as labels, keys(n) as keys"
    res = db.run_query_with_retry(query_labels)
    label_map = {}
    for r in res:
        l_tuple = tuple(sorted(r['labels']))
        if l_tuple not in label_map:
            label_map[l_tuple] = set()
        label_map[l_tuple].update(r['keys'])
    
    for labels, keys in label_map.items():
        print(f"Node {labels}: {sorted(list(keys))}")

    # 2. Relationships
    print("\n--- Relationships ---")
    query_rels = "MATCH (n)-[r]->(m) RETURN labels(n) as src, type(r) as rel, labels(m) as tgt"
    res = db.run_query_with_retry(query_rels)
    unique_rels = set()
    for r in res:
        src = "|".join(r['src']) if r['src'] else "Unknown"
        tgt = "|".join(r['tgt']) if r['tgt'] else "Unknown"
        unique_rels.add(f"({src}) -[:{r['rel']}]-> ({tgt})")
    
    for rel in sorted(list(unique_rels)):
        print(rel)

if __name__ == '__main__':
    discover()
