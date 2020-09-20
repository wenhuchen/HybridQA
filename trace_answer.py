import json

def loop_through_table(answer, table, results, matched_cells):
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            assert isinstance(cell[0], str)
            if answer.lower() == cell[0].lower():
                results.append((cell[0], (i, j), None, 'table'))
                matched_cells.append((i, j))
            elif " " + answer.lower() + " " in " " + cell[0].lower() + " ":
                results.append((cell[0], (i, j), None, 'table'))
                matched_cells.append((i, j))
            else:
                pass

def searchForAnswer(answer, table, passages, mapping_entity):
    results, matched_cells = [], []
    loop_through_table(answer, table, results, matched_cells)

    for k, v in passages.items():
        if k in mapping_entity:
            if " " + answer.lower() + " " in " " + v.lower() + " ":
                for loc in mapping_entity[k]:
                    if loc not in matched_cells:
                        results.append((k.replace('/wiki/', '').replace('_', ' '), loc, k, 'passage'))
        else:
            continue

    return results

def trace_answer_node(d):
    results = []
    table_id = d['table_id']
    
    with open('WikiTables-WithLinks/tables_tok/{}.json'.format(table_id)) as f:
        table = json.load(f) 
    with open('WikiTables-WithLinks/request_tok/{}.json'.format(table_id)) as f:
        requested_documents = json.load(f)   
    
    mapping_entity = {}
    for row_idx, row in enumerate(table['data']):
        for col_idx, cell in enumerate(row):
            for i, ent in enumerate(cell[1]):
                mapping_entity[ent] = mapping_entity.get(ent, []) + [(row_idx, col_idx)]

    results = searchForAnswer(d['answer-text'], table['data'], requested_documents, mapping_entity)
    
    return results

for split in ['train', 'dev']:
    with open(f'released_data/{split}.json', 'r') as f:
        data = json.load(f)

    for d in data:
        answer_nodes = trace_answer_node(d)
        d['answer-node'] = answer_nodes

    with open(f'released_data/{split}.traced.json', 'w') as f:
        data = json.dump(data, f, indent=2)
