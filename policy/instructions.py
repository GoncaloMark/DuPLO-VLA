# Task instruction templates with variations for data augmentation
TASK_INSTRUCTIONS = {
    'pick-place': [
        "pick up the puck and place it in the goal",
        "grasp the object and move it to the target location",
        "pick up the red puck and put it in the goal position",
        "grab the puck and place it at the goal",
    ],
    'door-open': [
        "open the door",
        "pull the door open",
        "open the cabinet door by pulling the handle",
        "grasp the handle and open the door",
    ],
    'drawer-open': [
        "open the drawer",
        "pull the drawer open",
        "slide the drawer out",
        "grasp the handle and open the drawer",
    ],
    'button-press': [
        "press the button",
        "push down on the button",
        "press the red button with the gripper",
        "push the button down",
    ],
    'reach': [
        "reach the target position",
        "move the gripper to the goal location",
        "reach to the target point",
        "move the end effector to the goal",
    ],
    'push': [
        "push the puck to the goal",
        "slide the object to the target location",
        "push the puck towards the goal position",
        "move the puck by pushing it to the target",
    ],
    'window-open': [
        "open the window",
        "slide the window open",
        "push the window to open it",
        "open the window by sliding it",
    ],
    'peg-insert-side': [
        "insert the peg into the hole",
        "put the peg in the side hole",
        "grasp the peg and insert it into the target hole",
        "place the peg into the insertion point",
    ],
    'basketball': [
        "put the ball in the basket",
        "place the basketball in the hoop",
        "pick up the ball and put it in the basket",
        "grasp the ball and drop it in the hoop",
    ],
    'lever-pull': [
        "pull the lever",
        "grasp the lever and pull it down",
        "pull down on the lever handle",
        "move the lever by pulling it",
    ],
    'hammer': [
        "hammer the nail",
        "use the hammer to hit the nail",
        "grasp the hammer and strike the nail",
        "hit the nail with the hammer",
    ],
    'box-close': [
        "close the box",
        "shut the box lid",
        "push the box closed",
        "close the box by pushing the lid down",
    ],
    'drawer-close': [
        "close the drawer",
        "push the drawer closed",
        "slide the drawer shut",
        "close the drawer by pushing it in",
    ],
    'door-close': [
        "close the door",
        "push the door closed",
        "shut the cabinet door",
        "close the door by pushing it",
    ],
    'shelf-place': [
        "place the object on the shelf",
        "put the item on the shelf",
        "grasp the object and place it on the upper shelf",
        "move the object to the shelf",
    ],
}

def get_instruction(task_name, variation=0):
    """
    Get instruction for a task
    
    Args:
        task_name: Task name (e.g., 'pick-place')
        variation: Which variation to use (0-3 usually)
    
    Returns:
        instruction: Natural language instruction string
    """
    if task_name not in TASK_INSTRUCTIONS:
        # Fallback: convert task name to instruction
        return f"perform the {task_name.replace('-', ' ')} task"
    
    instructions = TASK_INSTRUCTIONS[task_name]
    return instructions[variation % len(instructions)]


def get_random_instruction(task_name):
    """Get a random instruction variation for a task"""
    import random
    if task_name not in TASK_INSTRUCTIONS:
        return f"perform the {task_name.replace('-', ' ')} task"
    
    instructions = TASK_INSTRUCTIONS[task_name]
    return random.choice(instructions)

TASK_SETS = {
    'starter': [
        'pick-place',
        'door-open', 
        'drawer-open',
        'button-press',
        'reach'
    ],
    
    'medium': [
        'pick-place',
        'door-open',
        'drawer-open', 
        'button-press',
        'reach',
        'push',
        'window-open',
        'peg-insert-side',
        'basketball',
        'lever-pull',
    ],
    
    'full': [
        'pick-place',
        'door-open',
        'drawer-open',
        'button-press',
        'reach',
        'push',
        'window-open',
        'peg-insert-side',
        'basketball',
        'lever-pull',
        'hammer',
        'box-close',
        'drawer-close',
        'door-close',
        'shelf-place',
    ]
}

if __name__ == "__main__":
    print("="*80)
    print("Metaworld Task Instructions")
    print("="*80)
    
    for task_set_name, tasks in TASK_SETS.items():
        print(f"\n{task_set_name.upper()} SET ({len(tasks)} tasks):")
        print("-"*80)
        for task in tasks:
            instruction = get_instruction(task, variation=0)
            print(f"  {task:20s} → {instruction}")