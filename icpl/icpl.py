import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai-proxy.com/v1")

import re
import subprocess
from pathlib import Path
import shutil
import time 
import itertools

from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *

ICPL_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{ICPL_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
# Loading all text prompts
prompt_dir = f'{ICPL_ROOT_DIR}/utils/prompts'
behavior_feedback = file_to_string(f'{prompt_dir}/behavior_feedback.txt')
code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
history_prompt = file_to_string(f'{prompt_dir}/history_prompt.txt')
history_prompt_only_good = file_to_string(f'{prompt_dir}/history_prompt_only_good.txt')

def save_history(history_reward_function_path, path1, history, path2):
    with open(path1, 'w') as file:
        for iteration in history_reward_function_path:
            for code_path in iteration:
                file.write(code_path + " ")
            file.write("\n")
    with open(path2, 'w') as file:
        file.write(history)

def load_history(path1, path2):
    history_reward_function_path = []
    with open(path1, 'r') as file:
        lines = file.read().strip().split('\n')
        for line in lines:
            history_reward_function_path.append(line.strip().split(' '))
    history = file_to_string(path2)
    return history_reward_function_path, history

def generate_difference(code1, code2, name1, name2, cfg):

    difference_initial_system = file_to_string(f'{prompt_dir}/difference_initial_system.txt')
    difference_prompt = file_to_string(f'{prompt_dir}/difference.txt')

    messages = [{"role": "system", "content": difference_initial_system},
                {"role": "user", "content": difference_prompt.format(reward_code_1=code1, reward_code_2=code2, task_description=cfg.env.description)}]
    
    response_cur = None
    total_token = 0
    total_completion_token = 0
    prompt_tokens = 0
    logging.info(f"Generating difference")

    for attempt in range(1000):
        try:
            response_cur = client.chat.completions.create(model=cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            n=1)
            break
        except Exception as e:
            logging.info(f"Attempt {attempt+1} failed with error: {e}")
            time.sleep(1)
    if response_cur is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()
        
    difference = response_cur.choices[0].message.content
    prompt_tokens = response_cur.usage.prompt_tokens
    total_completion_token += response_cur.usage.completion_tokens
    total_token += response_cur.usage.total_tokens

    # Logging Token Information
    logging.info(f"Generating difference: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
    
    # replace first/second by name
    pattern_head_first = ["The first", "the first", "First", "first"]
    pattern_head_second = ["The second", "the second", "Second", "second"]
    pattern_tail = ["reward function", "function", "snippet", "code", "one"]
    
    for head in pattern_head_first:
        for tail in pattern_tail:
            difference = difference.replace(head + " " + tail, name1)

    for head in pattern_head_second:
        for tail in pattern_tail:
            difference = difference.replace(head + " " + tail, name2)

    return difference

def get_ordinal(i):
    if i == 11 or i == 12 or i == 13:
        return f"{i}th"
    if i % 10 == 1:
        return f"{i}st"
    if i % 10 == 2:
        return f"{i}nd"
    if i % 10 == 3:
        return f"{i}rd"
    return f"{i}th"

def generate_one_feedback(behavior_prompt, human_feedback, env_feedback):

    best_content = ""

    if env_feedback is not None:
        best_content += env_feedback
    if human_feedback is not None:
        best_content += behavior_prompt + human_feedback 

    best_content += "\n"
    return best_content 

def generate_specify_difference(history_reward_function_path, cfg):

    difference = ""
    i = len(history_reward_function_path)-1

    if cfg.use_reward_trace:
        if i > 0:
            reward_trace = file_to_string(history_reward_function_path[i-1][-1]).format(response_index=f"iter{i}-good")
            difference += reward_trace

    if i > 0:
        name1, name2 = f"iter{i+1}-good", f"iter{i}-good"
        code1, code2 = file_to_string(history_reward_function_path[i][0]), file_to_string(history_reward_function_path[i-1][0])
        difference += f"The difference between {name1} and {name2} is: \n {generate_difference(code1, code2, name1, name2, cfg)}\n"
    
    return difference

def generate_conversation(gpt_response_list, human_feedback_list, human_preference, env_feedback_list, history_reward_function_path, history, cfg):

    assistant_list = []
    user_list = []
    difference = ""

    # generate difference
    if history_reward_function_path is not None:
        
        if cfg.preference_type == 'only_good':
            if len(history_reward_function_path) > 1:
                difference += history_prompt_only_good.format(num_iterations=len(history_reward_function_path))
                difference += history
                difference += f"Next, a good example of reward functions generated in the {get_ordinal(len(history_reward_function_path))} iteration, i.e., iter{len(history_reward_function_path)}-good, is provided.\n"
        
        else:
            difference += history_prompt.format(num_iterations=len(history_reward_function_path))
            difference += history
            if len(gpt_response_list) == 2:
                difference += f"Next, the two reward functions generated in the {get_ordinal(len(history_reward_function_path))} iteration are provided.\n"
            else:
                difference += f"Next, the reward functions generated in the {get_ordinal(len(history_reward_function_path))} iteration are provided.\n"
        
    # generate feedback
    for i in range(len(gpt_response_list)):

        response_index = "the provided" if len(gpt_response_list) == 1 else f"the {get_ordinal(i+1)}"
        if human_feedback_list is not None:
            human_feedback = human_feedback_list[i]
        else:
            human_feedback = None
        if env_feedback_list is not None:
            env_feedback = env_feedback_list[i].format(response_index=response_index)
        else:
            env_feedback = None
        user_list.append(generate_one_feedback(behavior_feedback.format(response_index=response_index), 
                              human_feedback, env_feedback))
    
    # generate gpt response
    for i in range(len(gpt_response_list)):
        if len(gpt_response_list) != 1:
            assistant_list.append(f"The {get_ordinal(i+1)} generated reward function is provided:\n" + gpt_response_list[i] + '\n')
        elif cfg.preference_type == 'only_good':
            if history_reward_function_path is not None:
                if len(history_reward_function_path) > 1:
                    assistant_list.append(gpt_response_list[i] + '\n')
                else:
                    assistant_list.append(f"A good example of reward function generated in the last iteration is provided:\n" + gpt_response_list[i] + '\n')
            else:
                assistant_list.append(f"A good example of reward function generated in the last iteration is provided:\n" + gpt_response_list[i] + '\n')
        else:
            assistant_list.append(gpt_response_list[i] + '\n')
    
    if human_preference is not None:
        user_list[-1] += human_preference + '\n'
    if env_feedback_list is not None:
        user_list[-1] += code_feedback
    user_list[-1] += code_output_tip

    if cfg.user_list_only:
        return None, difference + "".join(list(itertools.chain.from_iterable(zip(assistant_list, user_list))))
    else:
        return "".join(assistant_list), difference + "".join(user_list)
        
def generate_reward_function(iter, num_samples, responses_buffer, messages, cfg):

    response_cur = None
    total_token = 0
    total_completion_token = 0
    prompt_tokens = 0
    chunk_size = num_samples*2 
    logging.info(f"Iteration {iter}: Generating {num_samples} samples with {cfg.model}")
    while True:
        if len(responses_buffer) >= num_samples:
            break
        for attempt in range(1000):
            try:
                response_cur = client.chat.completions.create(model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                n=chunk_size)
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()
            
        responses_buffer.extend(response_cur.choices)
        prompt_tokens = response_cur.usage.prompt_tokens
        total_completion_token += response_cur.usage.completion_tokens
        total_token += response_cur.usage.total_tokens
        
    # Logging Token Information
    logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        

def process_reward_function(iter, responses_id_list, responses, code_runs, rl_runs, output_file, task_code_string, task, suffix, cfg):
    
    regenerate_list = []
    failed = [] 
    for response_id in responses_id_list:
        response_cur = responses[response_id].message.content
        logging.info(f"Iteration {iter}: Processing Code Run {response_id}")
        with open(f"GPToutput_iter{iter}_response{response_id}.txt", 'w') as file:
            file.write(response_cur)
        # Regex patterns to extract python code enclosed in GPT response
        patterns = [
            r'```python(.*?)```', # 懒惰模式（非贪婪模式），获取最短的能满足条件的字符串。
            r'```(.*?)```',
            r'"""(.*?)"""',
            r'""(.*?)""',
            r'"(.*?)"',
        ]
        for pattern in patterns:
            code_string = re.search(pattern, response_cur, re.DOTALL)
            if code_string is not None:
                code_string = code_string.group(1).strip()
                break
        code_string = response_cur if not code_string else code_string

        # Remove unnecessary imports
        lines = code_string.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                code_string = "\n".join(lines[i:])
                
        # Add the ICPL Reward Signature to the environment code
        try:
            gpt_reward_signature, input_lst = get_function_signature(code_string)
        except Exception as e:
            logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
            regenerate_list.append(response_id)
            failed_reason = f"{e}"
            failed.append((response_cur, failed_reason))
            continue 

        code_runs[response_id] = code_string
        reward_signature = [
            f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
            f"self.extras['gpt_reward'] = self.rew_buf.mean()",
            f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
        ]
        indent = " " * 8
        reward_signature = "\n".join([indent + line for line in reward_signature])
        if "def compute_reward(self)" in task_code_string:
            task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
        elif "def compute_reward(self, actions)" in task_code_string:
            task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
        else:
            raise NotImplementedError

        # Save the new environment code when the output contains valid code string!
        with open(output_file, 'w') as file:
            file.writelines(task_code_string_iter + '\n')
            file.writelines("from typing import Tuple, Dict" + '\n')
            file.writelines("import math" + '\n')
            file.writelines("import torch" + '\n')
            file.writelines("from torch import Tensor" + '\n')
            if "@torch.jit.script" not in code_string:
                code_string = "@torch.jit.script\n" + code_string
            file.writelines(code_string + '\n')

        with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
            file.writelines(code_string + '\n')

        # Copy the generated environment code to hydra output directory for bookkeeping
        shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

        # Find the freest GPU to run GPU-accelerated RL
        set_freest_gpu()
        
        # Execute the python file with flags
        rl_filepath = f"env_iter{iter}_response{response_id}.txt"
        with open(rl_filepath, 'w') as f:
            f.writelines(file_to_string(output_file))
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                        f'max_iterations={cfg.max_iterations}'],
                                        stdout=f, stderr=f)
        success_run = block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
        
        if not success_run:
            regenerate_list.append(response_id)
            traceback = filter_traceback(file_to_string(rl_filepath)).split('\n')
            failed_reason = ""
            for line in reversed(traceback):
                if line:
                    failed_reason = line
                    break
            failed.append((response_cur, failed_reason))
            continue
        rl_runs[response_id] = process
    
    return regenerate_list, failed
    
@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ICPL_ROOT_DIR}")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac'
    task_file = f'{ICPL_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{ICPL_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    task_file_inner_rew_only = f'{ICPL_ROOT_DIR}/envs/{env_parent}/{env_name}_inner_reward_only.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    
    logging.info(f"Initial user:\n" + initial_user + "\n")

    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_string = task_code_string.replace(task, task+suffix)
    # Create Task YAML files
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    history_reward_function_path = []
    history = ""
    
    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # ICPL generation loop
    for iter in range(cfg.iteration):

        # Get ICPL response
        responses = [None] * cfg.sample
        code_runs = [None] * cfg.sample
        rl_runs = [None] * cfg.sample
        responses_id_list = list(range(0, cfg.sample)) # (re)generate failed responses
        responses_buffer = []
        failed_message = []

        attempt = 0
        while len(responses_id_list) > 0:
            logging.info(f"Iteration {iter}: (Re)generating samples with ids {responses_id_list}")
            generate_reward_function(iter, len(responses_id_list), responses_buffer, messages, cfg)
            for i, id in enumerate(responses_id_list):
                responses[id] = responses_buffer[i]
            responses_buffer = responses_buffer[len(responses_id_list):]
            responses_id_list, failed = process_reward_function(iter, responses_id_list, responses, code_runs, rl_runs, output_file, task_code_string, task, suffix, cfg)
            attempt += 1
            if attempt >= 200:
                logging.info(f"Finish due to cannot generate runable code")
                exit(0)
            
        # Gather RL training results and construct reward reflection
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        render_runs = []
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)
                
                content += policy_feedback.format(epoch_freq=epoch_freq, response_index=r"{response_index}")
                
                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric 
                            else:
                                metric_name = "task_score"
                            if cfg.use_sparse_reward or metric_name != "task_score":
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in tensorboard_logs and cfg.use_sparse_reward:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                
                with open(f"env_feedback_iter{iter}_response{response_id}.txt", 'w') as file:
                    file.write(content)

                content += code_feedback + '\n'  # Here do not save prompt into file

                if cfg.render:
                    # Render video
                    for i, line in enumerate(lines):
                        if line.startswith('Network Directory:'):
                            break 
                    network_logdir = line.split(':')[-1].strip() 
                    
                    for i, line in enumerate(reversed(lines)):
                        if line.startswith('=> saving checkpoint'):
                            break 
                    checkpoint_name = line.split('/')[-1].strip()[:-1]
                    
                    # checkpoint_name = task + suffix + '.pth'
                    checkpoint = os.path.join(network_logdir, checkpoint_name)
                    logging.info(f"Iteration {iter}: Run {response_id} load checkpoint from {checkpoint}")

                    shutil.copy(f"env_iter{iter}_response{response_id}.py", output_file)
                    render_filepath = f"reward_code_render_iter{iter}_response{response_id}.txt"
                    with open(render_filepath, 'w') as f:
                        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                                    f'hydra.run.dir=./render_iter{iter}_response{response_id}',
                                                    f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                                    f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                                    f'headless=False', f'capture_video=True', 'force_render=False', f'seed=0',
                                                    f'test=True',f'checkpoint={checkpoint}', f'num_envs={cfg.num_envs}',
                                                    ],
                                                    stdout=f, stderr=f)
                        
                    # Ensure that the RL training has started before moving on
                    while True:
                        render_log = file_to_string(render_filepath)
                        if "steps:" in render_log or "Traceback" in render_log:
                            if "steps:" in render_log:
                                logging.info(f"Iteration {iter}: Code Run {response_id} successfully rendered!")
                            if "Traceback" in render_log:
                                logging.info(f"Iteration {iter}: Code Run {response_id} execution error when rendering!")
                            break

                    render_runs.append(process)
                    process.communicate()

            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            contents.append(content) 
        
        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue
            
        for render_run in render_runs:
            render_run.communicate()

        if cfg.preference_type != 'simple_human': 
            logging.info(f"Iteration {iter}: Successes: {successes}")
        # Select the code samples based on human

        # generate message
        if cfg.preference_type == 'simple_human':
            while True:
                sample_idx_str = input("Please enter two number like \"x y\", \
where x is the index of the most preferred video and y is the index of the least preferred video according to the task description (0-indexed).\n")
                    
                try:
                    pair = sample_idx_str.strip().split()
                    assert len(pair) == 2
                    assert int(pair[0]) != int(pair[1])
                    for x in pair:
                        assert 0 <= int(x) and int(x) < cfg.sample
                    break
                except Exception as e:
                    print(f"Failed with error {e}, please try again.")

        elif cfg.preference_type == 'auto':
            good = np.argmax(np.array(successes))
            successes2 = []
            for i in successes:
                if i <= -10000:
                    successes2.append(10000)
                else:
                    successes2.append(i)
            bad = np.argmin(np.array(successes2))
            sample_idx_str = str(good) + ' ' + str(bad)

        elif cfg.preference_type in ['no', 'only_good']:
            sample_idx_str = str(np.argmax(np.array(successes)))
        else:
            raise NotImplementedError

        gpt_response_list = []
        human_feedback_list = []
        env_feedback_list = []
        code_dirs = []
        iter_idx_1 = None
        response_idx_1 = None
        for pairs in sample_idx_str.strip().split():
            if len(pairs.split('-')) == 1:
                iter_idx = iter
                response_idx = int(pairs)
            else:
                iter_idx = int(pairs.split('-')[0])
                response_idx = int(pairs.split('-')[1])
            
            if response_idx_1 is None:
                iter_idx_1 = iter
                response_idx_1 = response_idx

            code_dirs.append(os.path.join(workspace_dir, f"env_iter{iter_idx}_response{response_idx}_rewardonly.py"))

            if cfg.preference_type == 'no':
                gpt_response_list.append(file_to_string(f"GPToutput_iter{iter_idx}_response{response_idx}.txt"))
            else:
                gpt_response_list.append(file_to_string(f"env_iter{iter_idx}_response{response_idx}_rewardonly.py"))
            
            if cfg.use_human_feedback:
                human_feedback_list.append(file_to_string(f"human_feedback_iter{iter_idx}_response{response_idx}.txt"))
            if cfg.use_env_feedback:
                env_feedback_list.append(file_to_string(f"env_feedback_iter{iter_idx}_response{response_idx}.txt"))
            
        if cfg.use_human_feedback:
            assert len(human_feedback_list) >= 1

        if not cfg.use_env_feedback:
            env_feedback_list = None
        if not cfg.use_human_feedback:
            human_feedback_list = None

        if cfg.preference_type in ['auto', 'simple_human']:
            human_preference = file_to_string(f"{prompt_dir}/preference.txt")
        elif cfg.preference_type in ['no', 'only_good']:
            human_preference = None
        else:
            raise NotImplementedError

        if cfg.use_history_diff:
            code_dirs.append(os.path.join(workspace_dir, f"env_feedback_iter{iter_idx_1}_response{response_idx_1}.txt"))
            history_reward_function_path.append(code_dirs)
            history += generate_specify_difference(history_reward_function_path, cfg)
            save_history(history_reward_function_path, f"history_reward_function_path{iter}.txt", history, f"history{iter}.txt")
            hrfp = history_reward_function_path
        else:
            hrfp = None

        gpt_response, best_content = generate_conversation(gpt_response_list, human_feedback_list, human_preference, env_feedback_list, hrfp, history, cfg)
        # generate message end
            
        if len(successes) == 0: # for humanoid_circle
            successes = np.zeros(cfg.sample)
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        # Update the best ICPL Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        if cfg.preference_type == 'simple_human': 
            logging.info(f"Iteration {iter}: Successes: {successes}")
        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        if not cfg.open_loop: 
            if gpt_response is not None:
                logging.info(f"Iteration {iter}: GPT Output Content:\n" + gpt_response + "\n")
            logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
                
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

        
        if not cfg.open_loop: 
            if len(messages) == 2:
                if not cfg.user_list_only:
                    messages += [{"role": "assistant", "content": gpt_response}]
                messages += [{"role": "user", "content": best_content}]
            else:
                if not cfg.user_list_only:
                    assert len(messages) == 4
                    messages[-2] = {"role": "assistant", "content": gpt_response}
                else:
                    assert len(messages) == 3
                messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    # Evaluate the best reward code many times
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"eval: {max_reward_code_path}")
    shutil.copy(max_reward_code_path, output_file)

    eval_runs = []
    for i in range(cfg.num_eval):
        logging.info(f"Running {i}-th")
        set_freest_gpu()
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                                        ],
                                        stdout=f, stderr=f)
        
        block_until_training(rl_filepath, log_status=True)
        eval_runs.append(process)
    logging.info(f"Waiting")

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)

if __name__ == "__main__":
    main()