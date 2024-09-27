import json

with open("data_lmflow/test_raw.json",'r') as f:
    data = json.load(f)
    
lmflow_format_data = {
    "type": "conversation",
    "instances": []
}
for i, sample in enumerate(data):
    new_format = {
        "system": "",
        "conversation_id": i,
        "messages": [
        ]
    }
    items = sample['items']
    for j in items:
        if j['from'] == "human":
            new_format['messages'].append({
                "role":"user",
                "content":j['value']
            })
        else:
            new_format['messages'].append({
                "role":"assistant",
                "content":j['value']
            })
    lmflow_format_data['instances'].append(new_format)
    
with open("data_lmflow/test.json",'w') as f:
    json.dump(lmflow_format_data,f,indent=4,ensure_ascii=False)