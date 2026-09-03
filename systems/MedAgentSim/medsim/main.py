import argparse
import os
import json
import time
import openai
from medsim.core.agent import MeasurementAgent, PatientAgent, DoctorAgent, compare_results
from medsim.core.scenario import *
from medsim.query_model import *

# MedPIGuard - Update 4: Add tqdm for progress bar
from tqdm import tqdm
# MedPIGuard - End of Update 4

# MedPIGuard - Integrate 1: import API for AgentClinic
import sys
sys.path.append("/home/jupyter-truongnn/MedPI-Guard/")

from schemes.medpiguard import AttackScenario, DefenseMechanism
# MedPIGuard - End integrate 1.


def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm,
         measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences,
         anthropic_api_key=None, 
         output_dir="scenario_results_history", 
         attack_scenario=None, injection_strategy=None, attack_scenario_variant=False,
         defense_mode=None, max_defense_retries=3):

    # MedPIGuard - Defense Mechanism Initialization
    if defense_mode is not None:
        defense_mechanism = DefenseMechanism(defense_mode)

    # MedPIGuard - Integrate 2: Initialize attack scenario and injection strategy
    n_payloads = 1
    if attack_scenario is not None and injection_strategy is not None:
        attacker = AttackScenario(attack_scenario)
        output_dir = os.path.join(output_dir, f"{dataset}_{attack_scenario}_{injection_strategy}")
        if attack_scenario_variant:
            n_payloads = attacker.n_variants
    else:
        attacker = None
        output_dir = os.path.join(output_dir, f"{dataset}")


    # # MedPIGuard - Update 3: Create output directory if it doesn't exist
    # output_dir = os.path.join(output_dir, f"{dataset}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    openai.api_key = api_key
    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or doctor_llm in replicate_llms:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # MedPIGuard - Update 3: Add output directory creation for saving scenario results
    # output_dir = "/media/mbzuaiser/SSD1/Komal/Documents/multiagent/MedAgentSim/output"
    os.makedirs(output_dir, exist_ok=True)
    # Load the appropriate scenario loader
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    # Resolve model names from aliases
    doctor_llm = resolve_model_name(doctor_llm)
    patient_llm = resolve_model_name(patient_llm)
    measurement_llm = resolve_model_name(measurement_llm)
    moderator_llm = resolve_model_name(moderator_llm)
    total_correct = 0
    total_presents = 0

    meas_agent = MeasurementAgent(backend_str=measurement_llm)
    patient_agent = PatientAgent(backend_str=patient_llm)
    doctor_agent = DoctorAgent(backend_str=doctor_llm)
    mpipe = BAgent(moderator_llm)

    # MedPIGuard - Update 0: Remove the print statement to avoid cluttering the output
    # print(f"Starting simulation with {num_scenarios} scenarios from dataset {dataset}.")

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

        # MedPIGuard - Update 4: Track the number of scenarios to run and ensure it does not exceed the available scenarios
    print("--------------------------------------------------------------")
    print("MedAgentSim PARAMETERS:")
    print(" - Inference Type: {}".format(inf_type))
    print(" - Doctor Bias: {}".format(doctor_bias))
    print(" - Patient Bias: {}".format(patient_bias))
    print(" - Doctor LLM: {}".format(doctor_llm))
    print(" - Patient LLM: {}".format(patient_llm))
    print(" - Measurement LLM: {}".format(measurement_llm))
    print(" - Moderator LLM: {}".format(moderator_llm))
    print(" - Agent Dataset: {}".format(dataset))
    print(" - Doctor Image Request: {}".format(img_request))
    print(" - Number of Scenarios: {}".format(num_scenarios))
    print(" - Total Inferences: {}".format(total_inferences))
    # MedPIGuard - Integrate 2: Print attack scenario and injection strategy if specified with red color
    if attacker is not None:
        print("\033[91m - Attack Scenario: {} with Injection Strategy: {}\033[0m".format(attack_scenario, injection_strategy))
        print("\033[91m - Attack Scenario Variant: {}\033[0m".format(attack_scenario_variant))
        print("\033[91m - Number of Payloads: {}\033[0m".format(n_payloads))
    # print defense mode if specified with blue color
    if defense_mode is not None:
        print("\033[94m - Defense Mode: {}\033[0m".format(defense_mode))
    # MedPIGuard - End Integrate 2
    print(" - Output Directory: {} with subdirectories: scenario_[id]".format(output_dir))
    print("--------------------------------------------------------------")

    # pbar = tqdm(total=total_inferences, unit="inference", ncols=0)
    # min_num_scenarios = min(num_scenarios, scenario_loader.num_scenarios)
    # MedPIGuard - End Update 4

    # MedPIGuard - Integrate 3: Loop through payloads and scenarios, allowing for multiple payloads if specified for variant attack scenarios
    for payload_index in range(n_payloads):
        # MedPIGuard - Update 4: Reset progress bar for each payload
        pbar = tqdm(total=total_inferences, unit="inference", ncols=0)
        min_num_scenarios = min(num_scenarios, scenario_loader.num_scenarios)

        for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
            # MedPIGuard - Defense Mechanism - Integrate 2: Reset terminal flag for each scenario
            terminal_flag = False

            # MedPIGuard - Integrate 2: Print scenario ID and attack scenario information if specified
            if attacker is not None:
                attacker.initialize()
                if attack_scenario_variant:
                    payload_id = attacker.set_payload_variant(payload_index)
                else:
                    attacker.set_original_payload()

            total_presents += 1
            pi_dialogue = ""
            # dialogue_history = []

            # MedPIGuard - Update 3: Initialize dialogue history and patient interaction string
            dialogue_history = []
            scenario_output_dir = os.path.join(output_dir, f"scenario_{_scenario_id}")
            os.makedirs(scenario_output_dir, exist_ok=True)
            dialogue_file = os.path.join(
                scenario_output_dir,
                "dialogue_history.json" if not attack_scenario_variant else f"dialogue_history_{payload_id}.json"
                # f"dialogue_history_{payload_index}.json" if payload_index else "dialogue_history.json",
            )
            if os.path.exists(dialogue_file):
                # print(f"Dialogue history for scenario {_scenario_id} already exists. Skipping scenario.")
                continue
            # MedPIGuard - End Update 3
            
            # Initialize scenario and agents
            scenario = scenario_loader.get_scenario(id=_scenario_id)

            meas_agent.update_scenario(
                scenario=scenario)
            patient_agent.update_scenario(
                scenario=scenario, 
                bias_present=patient_bias)
            doctor_agent.update_scenario(
                scenario=scenario, 
                bias_present=doctor_bias,
                max_infs=total_inferences, 
                img_request=img_request)
            doctor_dialogue = ""    

            # MedPIGuard - Update 4: Reset progress bar for each scenario
            pbar.reset(total=total_inferences)
            pbar_description = f"{attack_scenario} + {injection_strategy}, " if attacker is not None else '' 
            pbar_description += f"Payload {payload_index+1}/{n_payloads}, " if attack_scenario_variant else ''
            pbar_description += f"Scenario {_scenario_id+1}/{min_num_scenarios}"
            pbar.set_description(pbar_description)

            for _inf_id in range(total_inferences):

                # MedPIGuard - Update 4: Use a while loop to allow for early termination if the doctor arrives at a diagnosis
                pbar.update(1)


                # Determine if images are requested
                if dataset == "NEJM":
                    imgs = "REQUEST IMAGES" in doctor_dialogue if img_request else True
                else:
                    imgs = False

                # Check if final inference
                if _inf_id == total_inferences - 1:
                    pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

                # Obtain doctor's dialogue
                if inf_type == "human_doctor":
                    doctor_dialogue = input("\nQuestion for patient: ")
                else:
                    doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)

                # Log and print the doctor's dialogue
                dialogue_text = f"Doctor [{int(((_inf_id+1)/total_inferences)*100)}%]: {doctor_dialogue}"
                # print(dialogue_text)   # MedPIGuard - Update 3: Remove print command to avoid cluttering the output
                dialogue_history.append({"speaker": "Doctor", "text": doctor_dialogue})

                # Check for diagnosis
                if "DIAGNOSIS READY" in doctor_dialogue or _inf_id == total_inferences:
                    correctness = compare_results(doctor_dialogue, scenario.diagnosis_information(), mpipe)
                    if correctness:
                        total_correct += 1
                    result_text = f"\nCorrect answer: {scenario.diagnosis_information()}"
                    scene_text = f"Scene {_scenario_id}, The diagnosis was {'CORRECT' if correctness else 'INCORRECT'} ({int((total_correct/total_presents)*100)}%)"
                    # print(result_text) # MedPIGuard - Update 3: Remove print command to avoid cluttering the output
                    # print(scene_text) # MedPIGuard - Update 3: Remove print command to avoid cluttering the output
                    # Add this scenario's conversation to the master log
                    dialogue_history.append(
                        {
                            "DIAGNOSIS_READY_Answer": scenario.diagnosis_information(),
                            "DIAGNOSIS_READY_Simulation": f"Scene {_scenario_id}, The diagnosis was "
                            f"{'CORRECT' if correctness else 'INCORRECT'}, "
                            f"{int((total_correct / total_presents) * 100)}%",
                        }
                    )
                    break

                # Handle medical exam request
                if "REQUEST TEST" in doctor_dialogue:
                    pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
                    measurement_text = f"Measurement [{int(((_inf_id+1)/total_inferences)*100)}%]: {pi_dialogue}"
                    # print(measurement_text) # MedPIGuard - Update 3: Remove print command to avoid cluttering the output
                    patient_agent.add_hist(pi_dialogue)
                    dialogue_history.append(
                        {"speaker": "Measurement", "text": pi_dialogue}
                    )
                else:
                    # Obtain patient's response
                    ### MedPIGuard - Defense Mechanism - Integrate 1: Comment old code
                    # if inf_type == "human_patient":
                    #     pi_dialogue = input("\nResponse to doctor: ")
                    # else:
                    #     pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                    # patient_text = f"Patient [{int(((_inf_id+1)/total_inferences)*100)}%]: {pi_dialogue}"
                    # # print(patient_text) # MedPIGuard - Update 3: Remove print command to avoid cluttering the output
                    # meas_agent.add_hist(pi_dialogue)
                    # dialogue_history.append({"speaker": "Patient", "text": pi_dialogue})

                    ### MedPIGuard - Defense Mechanism - Integrate 1: New code with defense mechanism
                    prompt_injection_flag = True
                    retries = 0
                    while prompt_injection_flag:
                        if inf_type == "human_patient":
                            pi_dialogue = input("\nResponse to doctor: ")
                        else:
                            pi_dialogue = patient_agent.inference_patient(doctor_dialogue)

                        # MedPIGuard - Integrate 2: Inject attack scenario into patient response if specified
                        if attacker is not None:
                            pi_dialogue = attacker.inject_payload(pi_dialogue, injection_strategy)

                        prompt_injection_flag = defense_mechanism.check_prompt_injection(pi_dialogue,doctor_context=doctor_dialogue)
                        if prompt_injection_flag:
                            retries += 1
                            # Emulate a safe response from the patient agent to replace the unsafe response
                            dialogue_history.append({"speaker": "Doctor", "text": "Dear patient, your response seems to contain unsafe or harmful content. Please provide a safe and appropriate response."})

                            if retries >= max_defense_retries:
                                # Emulate Termination by Defense Mechanism due to repeated unsafe responses
                                dialogue_history.append({"DIAGNOSIS_READY_Error": "TERMINAL by DEFENSE: Patient failed to provide a safe response after multiple attempts."})

                                # Terminate flag to break the inference loop
                                prompt_injection_flag = False
                                terminal_flag = True

                        dialogue_history.append({"speaker": "Patient", "text": pi_dialogue})

                    meas_agent.add_hist(pi_dialogue)

                # Prevent API timeouts
                time.sleep(1.0)

                # MedPIGuard - Defense Mechanism - Update 4: Break the inference loop if the doctor has arrived at a diagnosis or if the defense mechanism has terminated the scenario
                if terminal_flag:
                    break
                
            
            # Save the dialogue history to a JSON file at the end of each scenario

            # MedPIGuard - Commented out the previous code for saving dialogue history to avoid overwriting in case of multiple payloads
            # scenario_output_dir = os.path.join(output_dir, f"scenario_{_scenario_id}")
            # os.makedirs(scenario_output_dir, exist_ok=True) 
            # dialogue_file = os.path.join(scenario_output_dir, "dialogue_history.json")
            with open(dialogue_file, "w") as f:
                json.dump(dialogue_history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='gpt4')
    parser.add_argument('--patient_llm', type=str, default='gpt4')
    parser.add_argument('--measurement_llm', type=str, default='gpt4')
    parser.add_argument('--moderator_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=20, required=False, help='Number of inferences between patient and doctor')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False, help='Anthropic API key for Claude 3.5 Sonnet')
    
    # MedPIGuard - Update 3: Add argument for output directory
    parser.add_argument('--output_dir', type=str, default="scenario_results_history", required=False, help='Directory to save scenario outputs, defaults to scenario_results_history')
    # MedPIGuard - Integrate 2: Add arguments for attack scenario and injection strategy in Attack Simulation
    parser.add_argument('--attack_scenario', type=str, default=None, choices=['T1', 'T2', 'T3', 'T4'], help='Attack scenario to test the model with') 
    parser.add_argument('--injection_strategy', type=str, default=None, help='Injection strategy for patient responses. Options:  ' \
                        'I1 (Single-turn Injection), ' \
                        'I2 (Multi-turn Injection)', choices=['I1', 'I2'])
    parser.add_argument('--attack_scenario_variant', action='store_true', help='Enable variant attack scenarios.')
    # MedPIGuard - Integrate 3: Add arguments for defense mode and max retries for unsafe patient responses
    parser.add_argument('--defense_mode', type=str, default='off',
                            choices=['off', 'regex', 'protectai', 'sentinel', 'sentinelv2', 'Instruction', 'commandsans', 'medpid'],
                            help='Defense mode for checking patient prompt injection')
    parser.add_argument('--max_defense_retries', type=int, default=3,
                        help='Max retries for unsafe patient response before forcing safe fallback')
    args = parser.parse_args()

    main(args.openai_api_key, args.replicate_api_key, args.inf_type, args.doctor_bias, args.patient_bias, args.doctor_llm, args.patient_llm, args.measurement_llm, args.moderator_llm, args.num_scenarios, args.agent_dataset, args.doctor_image_request, args.total_inferences, args.anthropic_api_key,
         args.output_dir, args.attack_scenario, args.injection_strategy, args.attack_scenario_variant, args.defense_mode, args.max_defense_retries)


## terminal running bash
# python medsim/main.py --inf_type llm --doctor_bias None --patient_bias None --doctor_llm meta-llama/Llama-3.3-70B-Instruct --patient_llm meta-llama/Llama-3.3-70B-Instruct --measurement_llm meta-llama/Llama-3.3-70B-Instruct --moderator_llm meta-llama/Llama-3.3-70B-Instruct --agent_dataset MedQA --doctor_image_request False --num_scenarios 10 --total_inferences 20
