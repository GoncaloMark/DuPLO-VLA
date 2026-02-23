import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env import MetaWorldEnv
from termcolor import cprint
import copy
from metaworld.policies import *

seed = np.random.randint(0, 100)

TASK_INSTRUCTIONS = {
    'pick-place': [
        "pick up the puck and place it in the goal",
        "grasp the object and move it to the target location",
        "pick up the red puck and put it in the goal position",
        "grab the puck and place it at the goal",
        "lift the puck and drop it on the goal marker",
        "transport the object from its current position to the target",
    ],
    'door-open': [
        "open the door",
        "pull the door open",
        "open the cabinet door by pulling the handle",
        "grasp the handle and open the door",
        "swing the door open by pulling its handle",
        "grip the door handle and pull it open",
    ],
    'drawer-open': [
        "open the drawer",
        "pull the drawer open",
        "slide the drawer out",
        "grasp the handle and open the drawer",
        "extend the drawer outward by pulling its handle",
        "pull the drawer all the way out",
    ],
    'button-press': [
        "press the button",
        "push down on the button",
        "press the red button with the gripper",
        "push the button down",
        "depress the button until it clicks",
        "apply downward force on the button to activate it",
    ],
    'reach': [
        "reach the target position",
        "move the gripper to the goal location",
        "reach to the target point",
        "move the end effector to the goal",
        "position the gripper at the marked target",
        "navigate the arm tip to the goal position",
    ],
    # 'assembly': [
    #     "insert the peg into the hole",
    #     "fit the peg into the hole by aligning and pushing it in",
    #     "pick up the peg and insert it into the target hole",
    #     "align the peg with the hole and press it in",
    #     "grasp the peg and seat it in the hole",
    #     "insert the cylindrical peg into the matching socket",
    # ],
    'basketball': [
        "dunk the ball into the basket",
        "pick up the basketball and drop it through the hoop",
        "lift the ball and place it into the basket",
        "grab the basketball and shoot it into the hoop",
        "grasp the ball and put it through the basket",
        "pick the ball up and drop it into the goal",
    ],
    'bin-picking': [
        "pick up the object from the bin",
        "grasp the item inside the bin and lift it out",
        "retrieve the object from the container",
        "reach into the bin and pick up the object",
        "lift the object out of the bin",
        "grab the item from inside the bin and pull it free",
    ],
    'box-close': [
        "close the box lid",
        "push the lid down to close the box",
        "grasp the lid and place it on top of the box",
        "shut the box by pressing the lid closed",
        "pick up the box lid and close the container",
        "fold the lid down onto the box to seal it",
    ],
    'button-press-topdown': [
        "press the button from above",
        "push the button downward with the gripper",
        "apply downward pressure to the button",
        "press the button using a top-down approach",
        "depress the button from directly above",
        "drive the gripper down onto the button",
    ],
    'button-press-topdown-wall': [
        "press the button from above near the wall",
        "push the button down while avoiding the wall",
        "approach from above and press the button beside the wall",
        "depress the button with a downward motion near the wall",
        "press the button from the top, navigating around the wall",
        "push the button down near the wall obstacle",
    ],
    'button-press-wall': [
        "press the button near the wall",
        "push the button while navigating around the wall",
        "reach past the wall and press the button",
        "depress the button located near the wall",
        "press the button beside the wall obstacle",
        "navigate around the wall and push the button",
    ],
    'coffee-button': [
        "press the coffee machine button",
        "push the button on the coffee maker",
        "activate the coffee machine by pressing its button",
        "press the start button on the coffee machine",
        "push the coffee maker's button to start it",
        "depress the button on the front of the coffee machine",
    ],
    'coffee-pull': [
        "pull the coffee mug",
        "slide the coffee mug toward you",
        "drag the mug across the surface",
        "pull the coffee cup to the target position",
        "move the mug by pulling it along the counter",
        "grasp the mug and pull it to the goal",
    ],
    'coffee-push': [
        "push the coffee mug",
        "slide the coffee mug forward",
        "push the mug to the target location",
        "shove the cup toward the goal position",
        "move the mug by pushing it along the surface",
        "apply force to the coffee mug to push it to the target",
    ],
    'dial-turn': [
        "turn the dial",
        "rotate the dial to the target position",
        "spin the dial clockwise to the goal",
        "grasp the dial and rotate it",
        "twist the dial to the marked position",
        "turn the knob until it reaches the target angle",
    ],
    # 'disassemble': [
    #     "pull the peg out of the hole",
    #     "remove the peg from the socket",
    #     "extract the peg from the hole by pulling upward",
    #     "grasp the peg and pull it free from the hole",
    #     "detach the peg from the assembly by pulling it out",
    #     "grip the peg and withdraw it from the socket",
    # ],
    'door-close': [
        "close the door",
        "push the door shut",
        "swing the door closed",
        "push the door back to its closed position",
        "apply force to the door to close it",
        "shut the door by pushing it closed",
    ],
    'door-lock': [
        "lock the door",
        "turn the lock to secure the door",
        "engage the door lock",
        "rotate the locking mechanism to lock the door",
        "grasp the lock and turn it to the locked position",
        "secure the door by turning the lock",
    ],
    'door-unlock': [
        "unlock the door",
        "rotate the lock to release the door",
        "disengage the door lock",
        "turn the locking mechanism to unlock the door",
        "grasp the lock and rotate it to the open position",
        "release the door lock by turning it",
    ],
    'drawer-close': [
        "close the drawer",
        "push the drawer closed",
        "slide the drawer back in",
        "push the drawer all the way in",
        "return the drawer to its closed position",
        "apply inward force to close the drawer",
    ],
    'faucet-close': [
        "close the faucet",
        "turn the faucet handle to the off position",
        "rotate the faucet to close it",
        "shut the faucet by turning the handle",
        "grasp the faucet handle and close it",
        "turn the tap off",
    ],
    'faucet-open': [
        "open the faucet",
        "turn the faucet handle to the on position",
        "rotate the faucet to open it",
        "turn the faucet on by rotating its handle",
        "grasp the faucet and twist it open",
        "turn the tap on",
    ],
    'hammer': [
        "hammer the nail",
        "strike the nail with the hammer",
        "use the hammer to drive the nail in",
        "hit the nail on the head with the hammer",
        "grab the hammer and drive the nail into the surface",
        "pound the nail down using the hammer",
    ],
    'handle-press': [
        "press the handle down",
        "push the handle downward",
        "apply downward force to the handle",
        "depress the handle until it reaches the bottom",
        "push the lever handle down to its end position",
        "drive the handle downward",
    ],
    'handle-pull': [
        "pull the handle",
        "grasp the handle and pull it toward you",
        "pull the handle up to its target position",
        "grip the handle and pull it outward",
        "tug the handle to the goal position",
        "extend the handle by pulling it",
    ],
    'lever-pull': [
        "pull the lever",
        "grasp the lever and pull it down",
        "move the lever to the target position by pulling",
        "grip the lever and pull it toward you",
        "actuate the lever by pulling it",
        "pull the lever arm to the goal angle",
    ],
    'pick-out-of-hole': [
        "pick the object out of the hole",
        "retrieve the object from the hole",
        "grasp the object and lift it out of the hole",
        "extract the item from the hole by picking it up",
        "reach into the hole, grab the object, and lift it out",
        "pull the object upward out of the hole",
    ],
    # 'pick-place-wall': [
    #     "pick and place the object over the wall",
    #     "lift the object over the wall and place it at the goal",
    #     "pick up the puck, clear the wall, and place it at the target",
    #     "grasp the object, bring it over the barrier, and set it down",
    #     "move the object from one side of the wall to the other",
    #     "carry the object over the wall obstacle to the goal",
    # ],
    'plate-slide': [
        "slide the plate",
        "push the plate along the surface to the goal",
        "slide the plate to the target position",
        "move the plate by sliding it forward",
        "push the flat plate to the goal location",
        "apply force to the plate and slide it to the target",
    ],
    'plate-slide-back': [
        "slide the plate back",
        "push the plate backward to the goal",
        "slide the plate in the reverse direction to the target",
        "move the plate back to its original position",
        "push the plate rearward to the goal location",
        "reverse-slide the plate to the target",
    ],
    'push': [
        "push the object to the goal",
        "slide the object forward to the target",
        "apply force to the object and push it to the goal",
        "move the puck to the target by pushing it",
        "shove the object to the marked goal position",
        "push the object across the surface to the target",
    ],
    # 'push-back': [
    #     "push the object back",
    #     "slide the object backward to the goal",
    #     "move the object to the behind position",
    #     "push the puck rearward to the target",
    #     "apply backward force to return the object to goal",
    #     "reverse-push the object to the target position",
    # ],
    # 'push-wall': [
    #     "push the object to the goal near the wall",
    #     "slide the object to the target while navigating the wall",
    #     "push the puck to the goal, avoiding the wall obstacle",
    #     "move the object past the wall to the goal",
    #     "shove the object around the wall to the target",
    #     "navigate the wall and push the object to the goal",
    # ],
    # 'reach-wall': [
    #     "reach to the target near the wall",
    #     "move the gripper to the goal position beside the wall",
    #     "navigate past the wall and reach the target",
    #     "position the gripper at the goal near the wall obstacle",
    #     "reach the target point while avoiding the wall",
    #     "move the end effector to the goal near the wall",
    # ],
    'shelf-place': [
        "place the object on the shelf",
        "pick up the object and set it on the shelf",
        "lift the object and place it on the elevated shelf",
        "grasp the item and put it on the shelf",
        "carry the object up and place it on the shelf surface",
        "move the object to the shelf and set it down",
    ],
    # 'stick-pull': [
    #     "use the stick to pull the object",
    #     "grab the stick and hook the object to pull it",
    #     "use the tool to drag the object toward the goal",
    #     "pick up the stick, hook it on the object, and pull",
    #     "employ the stick as a tool to pull the object to the target",
    #     "use the elongated tool to drag the object closer",
    # ],
    # 'stick-push': [
    #     "use the stick to push the object",
    #     "grab the stick and push the object to the goal",
    #     "use the tool to shove the object forward",
    #     "pick up the stick and use it to move the object",
    #     "employ the stick to push the object to the target",
    #     "use the elongated tool to push the object to the goal",
    # ],
    'sweep': [
        "sweep the object to the goal",
        "brush the object along the surface to the target",
        "sweep the puck to the goal position",
        "push the object in a sweeping motion to the target",
        "move the object by sweeping it across the surface",
        "use a wide motion to sweep the object to the goal",
    ],
    'sweep-into': [
        "sweep the object into the hole",
        "brush the object into the opening",
        "sweep the puck into the goal hole",
        "push the object with a sweeping motion into the hole",
        "move the object by sweeping it into the target hole",
        "guide the object into the hole by sweeping",
    ],
    'window-close': [
        "close the window",
        "slide the window shut",
        "push the window to the closed position",
        "move the window panel to close the opening",
        "slide the window frame closed",
        "shut the window by sliding it closed",
    ],
    'window-open': [
        "open the window",
        "slide the window open",
        "pull the window to the open position",
        "move the window panel to open the gap",
        "slide the window frame open",
        "open the window by sliding it",
    ],
}

TEST_TASKS = {
    'stick-pull', 'stick-push',
    'reach-wall', 'push-wall', 'pick-place-wall',
    'assembly', 'disassemble',
}

TRAIN_TASKS= {k for k in TASK_INSTRUCTIONS if k not in TEST_TASKS}

def get_instruction(task_name, variation=0):
    instructions = TASK_INSTRUCTIONS.get(task_name, [f"perform the {task_name.replace('-', ' ')} task"])
    return instructions[variation % len(instructions)]


def load_mw_policy(task_name):
    if task_name == 'peg-insert-side':
        return SawyerPegInsertionSideV2Policy()
    parts = task_name.split('-')
    cls_name = "Sawyer" + "".join(p.capitalize() for p in parts) + "V2Policy"
    return eval(cls_name)()


def main(args, env_name):
    save_dir = os.path.join(args.root_dir, 'metaworld_' + env_name + '_expert.zarr')
    if os.path.exists(save_dir):
        cprint(f'Data already exists at {save_dir}', 'red')
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = 'n'
        if user_input == 'y':
            cprint(f'Overwriting {save_dir}', 'red')
            os.system(f'rm -rf {save_dir}')
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)

    e = MetaWorldEnv(env_name, device="cuda:0", use_point_crop=True)

    num_episodes = args.num_episodes
    cprint(f"Number of episodes : {num_episodes}", "yellow")
    cprint(f"Task: {env_name}", "yellow")

    total_count = 0
    img_arrays          = []
    point_cloud_arrays  = []
    depth_arrays        = []
    state_arrays        = []
    full_state_arrays   = []
    action_arrays       = []
    instruction_arrays  = []
    task_name_arrays    = []   
    episode_ends_arrays = []
    episode_id_arrays = []

    episode_idx = 0
    mw_policy   = load_mw_policy(env_name)

    while episode_idx < num_episodes:
        raw_state = e.reset()['full_state']
        obs_dict  = e.get_visual_obs()

        ep_reward        = 0.
        ep_success       = False
        ep_success_times = 0

        img_arrays_sub         = []
        point_cloud_arrays_sub = []
        depth_arrays_sub       = []
        state_arrays_sub       = []
        full_state_arrays_sub  = []
        action_arrays_sub      = []
        total_count_sub        = 0

        # Cycle through phrasing variants across episodes for diversity
        instruction = get_instruction(env_name, variation=episode_idx)
        cprint(f"Episode {episode_idx} instruction: '{instruction}'", "cyan")

        done = False
        while not done:
            total_count_sub += 1

            obs_img         = obs_dict['image']
            obs_robot_state = obs_dict['agent_pos']
            obs_point_cloud = obs_dict['point_cloud']
            obs_depth       = obs_dict['depth']

            img_arrays_sub.append(obs_img)
            point_cloud_arrays_sub.append(obs_point_cloud)
            depth_arrays_sub.append(obs_depth)
            state_arrays_sub.append(obs_robot_state)
            full_state_arrays_sub.append(raw_state)

            action = mw_policy.get_action(raw_state)
            action_arrays_sub.append(action)

            obs_dict, reward, done, info = e.step(action)
            raw_state         = obs_dict['full_state']
            ep_reward        += reward
            ep_success        = ep_success or info['success']
            ep_success_times += info['success']

        if not ep_success or ep_success_times < 5:
            cprint(f'Episode {episode_idx} failed — reward {ep_reward}, success_times {ep_success_times}', 'red')
            continue

        total_count += total_count_sub
        episode_ends_arrays.append(copy.deepcopy(total_count))

        img_arrays.extend(copy.deepcopy(img_arrays_sub))
        point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
        depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
        state_arrays.extend(copy.deepcopy(state_arrays_sub))
        action_arrays.extend(copy.deepcopy(action_arrays_sub))
        full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))

        # Both columns repeated for every timestep in the episode
        instruction_arrays.extend([instruction] * total_count_sub)
        task_name_arrays.extend([env_name] * total_count_sub)    
        episode_id_arrays.extend([episode_idx] * total_count_sub)

        cprint(
            f'Episode {episode_idx} — reward {ep_reward}, '
            f'success_times {ep_success_times}, instruction "{instruction}"', 'green'
        )
        episode_idx += 1

    img_arrays         = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3:   # (N, C, H, W) → (N, H, W, C)
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    state_arrays       = np.stack(state_arrays, axis=0)
    full_state_arrays  = np.stack(full_state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays       = np.stack(depth_arrays, axis=0)
    action_arrays      = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    instruction_arrays  = np.array(instruction_arrays, dtype='object')
    task_name_arrays    = np.array(task_name_arrays,   dtype='object')  
    episode_id_arrays = np.array(episode_id_arrays, dtype=np.int64)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    zarr_data.create_dataset('img',         data=img_arrays,
        chunks=(100, *img_arrays.shape[1:]),         dtype='uint8',   overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state',       data=state_arrays,
        chunks=(100, state_arrays.shape[1]),         dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('full_state',  data=full_state_arrays,
        chunks=(100, full_state_arrays.shape[1]),    dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays,
        chunks=(100, *point_cloud_arrays.shape[1:]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth',       data=depth_arrays,
        chunks=(100, *depth_arrays.shape[1:]),       dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action',      data=action_arrays,
        chunks=(100, action_arrays.shape[1]),        dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('instruction', data=instruction_arrays.astype(str),
        chunks=(100,), dtype=str, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('task_name',   data=task_name_arrays.astype(str),
        chunks=(100,), dtype=str, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('episode_id', data=episode_id_arrays,
        chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays,
        dtype='int64', overwrite=True, compressor=compressor)

    cprint(f'-' * 50, 'cyan')
    cprint(f'img shape:         {img_arrays.shape}', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}', 'green')
    cprint(f'state shape:       {state_arrays.shape}', 'green')
    cprint(f'action shape:      {action_arrays.shape}', 'green')
    cprint(f'instruction shape: {instruction_arrays.shape}', 'green')
    cprint(f'task_name shape:   {task_name_arrays.shape}', 'green')        
    cprint(f'Unique tasks:      {np.unique(task_name_arrays).tolist()}', 'cyan')  
    cprint(f'Saved zarr to {save_dir}', 'green')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes',  type=int, default=20)
    parser.add_argument('--root_dir',      type=str, default="data/")
    args = parser.parse_args()

    for env in TASK_INSTRUCTIONS.keys():
        main(args, env)
