import json

def convert_alpaca_json(input_json):
    output_json = []
    for item in input_json:
        conversation = [
            {"from": "human", "value": item["input"]},
            {"from": "gpt", "value": item["output"]}
        ]
        new_item = {
            "id": "alpaca-tw_en-align",
            "conversations": conversation,
            "instruction": item["instruction"]
        }
        output_json.append(new_item)
    return output_json

if __name__ == "__main__":
    input_file_path = "./data/alpaca-tw_en-align.json"
    output_file_path = "./data/alpaca-tw_en-align_converted.json"

    with open(input_file_path, "r", encoding="utf-8") as f:
        input_json_data = json.load(f)

    output_json_data = convert_alpaca_json(input_json_data)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, ensure_ascii=False, indent=4)

    print("JSON conversion completed. Output written to", output_file_path)
