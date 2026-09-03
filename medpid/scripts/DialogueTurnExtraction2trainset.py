import os, json, argparse
def extract_dialogue_turns(input_file, schedule="I3"):
    with open(input_file, "r") as f:
        dialogue_history = json.load(f)
    lst = []
    turn_patient_count = 0
    for i in range(len(dialogue_history)):
        if "speaker" in dialogue_history[i] and dialogue_history[i]['speaker'] == "Patient":
            turn_patient_count += 1
            if schedule == "I3":
                lst.append({
                    "doctor_question": dialogue_history[i-1]['text'] if i-1 >= 0 else "",
                    "patient_answer": dialogue_history[i]['text'],
                    "label": 1 if turn_patient_count < 3 else 0
                })
            elif schedule == "I1":
                lst.append({
                    "doctor_question": dialogue_history[i-1]['text'] if i-1 >= 0 else "",
                    "patient_answer": dialogue_history[i]['text'],
                    "label": 1 if turn_patient_count == 1 else 0
                })
    return lst

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Evaluate Judge on MedAgentSim attack scenarios.")
    args.add_argument(
        "--results_dir",
        type=str,
        default="data"
    )
    args = args.parse_args()
    all_extracted_turns = list()  # initialize an empty list to store all extracted turns across dialogue systems
    dialogue_history_dir = "/home/jupyter-truongnn/MedPI-Guard/ResultForPaper/RQ1_AttackEffectiveness"
    for dialogue_system in os.listdir(dialogue_history_dir):
        dialogue_system_path = os.path.join(dialogue_history_dir, dialogue_system)
        if not os.path.isdir(dialogue_system_path):
            continue
        system_extracted_turns = []
        for root, dirs, files in os.walk(dialogue_system_path):
            for file in files:
                if file.startswith("dialogue_history") and file.endswith(".json"):
                    input_file = os.path.join(root, file)
                    if "_I2/scenario" in input_file:
                        continue  # skip files in the I2 scenario
                    if "_I1/scenario" in input_file:
                        extracted_turns = extract_dialogue_turns(input_file, schedule="I1")
                    if "_I3/scenario" in input_file:
                        extracted_turns = extract_dialogue_turns(input_file, schedule="I3")
                    
                    system_extracted_turns.extend(extracted_turns)

        output_file = os.path.join(args.results_dir, f"train_dialogue_turns_{dialogue_system}.json")
        # remove duplicates
        unique_turns = {json.dumps(turn): turn for turn in system_extracted_turns}.values()
        with open(output_file, "w") as f:
            json.dump(list(unique_turns), f, indent=4)
        print(f"Extracted dialogue turns saved to {output_file}")

        # print phân bố label
        label_counts = {}
        for turn in unique_turns:
            label = turn['label']
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} turns")  
        
        all_extracted_turns += unique_turns  # add the extracted turns for this dialogue system to the overall list

    
    # save all_extracted_turns to a single CSV file
    # remove duplicates from all_extracted_turns
    all_extracted_turns = {json.dumps(turn): turn for turn in all_extracted_turns}.values()
    import pandas as pd
    df = pd.DataFrame(list(all_extracted_turns))
    df.to_csv(os.path.join(args.results_dir, "trainset.csv"), index=False)
    print(f"Extracted dialogue turns saved to {os.path.join(args.results_dir, 'trainset.csv')}")

    # print phân bố label cho toàn bộ tập dữ liệu
    label_counts = {}
    for turn in all_extracted_turns:
        label = turn['label']
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    print("Label distribution for the entire dataset:") 
    for label, count in label_counts.items():
          print(f"Label {label}: {count} turns")

    
    
    