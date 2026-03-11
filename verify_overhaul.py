from database import db

def verify():
    queries = [
        # Admission Requirement check
        ("MATCH (p:Program) WHERE toLower(p.Name) CONTAINS 'computer science' RETURN p.Name AS Program, p.Admission_Requirement AS Requirements", "Admission Requirements"),
        # Insight check
        ("MATCH (i:Insight)-[:PART_OF]->(p:Program) WHERE toLower(p.Name) CONTAINS 'nurse' RETURN i.Fact_Description LIMIT 1", "Nursing Insight"),
        # Campus Life check
        ("MATCH (caf:Cafeteria) RETURN caf.Name LIMIT 1", "Cafeteria check")
    ]
    
    for q, label in queries:
        try:
            res = db.run_query_with_retry(q)
            print(f"[{label}] Result: {res}")
        except Exception as e:
            print(f"[{label}] FAILED: {e}")

if __name__ == '__main__':
    verify()
