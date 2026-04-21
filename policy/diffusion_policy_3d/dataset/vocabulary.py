TASKS = [
    'pick-place', 'dial-turn', 'door-close', 'faucet-open', 'handle-pull'
]

INSTRUCTIONS = [
    "pick up the puck and place it in the goal",
    "grasp the object and move it to the target location",
    "pick up the red puck and put it in the goal position",
    "grab the puck and place it at the goal",
    "lift the puck and drop it on the goal marker",
    "transport the object from its current position to the target",
    "turn the dial",
    "rotate the dial to the target position",
    "spin the dial clockwise to the goal",
    "grasp the dial and rotate it",
    "twist the dial to the marked position",
    "turn the knob until it reaches the target angle",
    "close the door",
    "push the door shut",
    "swing the door closed",
    "push the door back to its closed position",
    "apply force to the door to close it",
    "shut the door by pushing it closed",
    "open the faucet",
    "turn the faucet handle to the on position",
    "rotate the faucet to open it",
    "turn the faucet on by rotating its handle",
    "grasp the faucet and twist it open",
    "turn the tap on",
    "pull the handle",
    "grasp the handle and pull it toward you",
    "pull the handle up to its target position",
    "grip the handle and pull it outward",
    "tug the handle to the goal position",
    "extend the handle by pulling it",
]

TASK_TO_ID = {task: i for i, task in enumerate(TASKS)}
INSTRUCTION_TO_ID = {inst: i for i, inst in enumerate(INSTRUCTIONS)}
