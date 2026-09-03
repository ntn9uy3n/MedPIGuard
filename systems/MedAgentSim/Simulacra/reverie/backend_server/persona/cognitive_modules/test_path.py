import os
import json
import glob
# current_file_dir = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.abspath(os.path.join(current_file_dir, "../../../../../medsim/configs/config_sim.yaml"))
# print("Current file dir:", config_path)
# print("Resolved config path:", os.path.abspath(config_path))
# def extract_sim_info():
#   current_file_dir = os.path.dirname(os.path.abspath(__file__))
#   json_file_path = os.path.join(current_file_dir, "../..", "simulation_controller.json")
#   json_file_path = os.path.normpath(json_file_path)  # Normalize the path

#   # Load the existing JSON file
#   with open(json_file_path, 'r') as file:
#     data = json.load(file)

#   # Get info
#   return data["total_scenarios"], data["total_correct"], data["num_scenarios"], data["simulation_index"]
# total_scenarios, total_correct, num_scenarios, idx = extract_sim_info()
# print(total_scenarios, total_correct, num_scenarios, idx)
# Get the directory of the current script
current_file_dir = os.path.dirname(os.path.abspath(__file__))
json_paths = os.path.abspath(os.path.join(current_file_dir, "../../../../..", "output"))
print(json_paths)
json_files = glob.glob(os.path.join(json_paths, "*.json"))
print(json_files)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
json_paths = os.path.abspath(os.path.join(current_file_dir, "../../../../..", "output"))
print("Output directory:", json_paths)
json_files = glob.glob(os.path.join(json_paths, "**", "*.json"), recursive=True)
print("JSON files found:", json_files)