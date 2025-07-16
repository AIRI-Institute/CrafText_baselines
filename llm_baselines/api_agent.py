from jinja2 import Template

from baselines.api_models.api_providers import APIProvider


class ApiAgent:
    default_available_actions = (
        "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP",
        "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT",
        "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
        "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD"
    )

    def __init__(self, api_name='dummy', prompt_template_filename="TEMPLATE.MD", available_actions=None,
                 dummy_mode=False):
        self.available_actions = available_actions or self.default_available_actions
        self.history = []
        self.action_history = []
        self.dummy_mode = dummy_mode
        self.step_counter = 0
        self.api_provider = APIProvider()
        self.api_name = api_name

        with open(prompt_template_filename, 'r') as f:
            self.prompt_template = Template(f.read())

    def act(self, text_observation, instruction):
        message_history = "\n".join(self.history[-4:]) if self.history else "None"
        action_history = ", ".join(self.action_history[-4:]) if self.history else "None"

        prompt = self.prompt_template.render(
            text_observation=text_observation,
            instruction=instruction,
            available_actions=self.available_actions,
            history=message_history,
            action_history=action_history,
            current_step =self.step_counter,
        )

        print('============== Prompt ==============')
        print(prompt)

        raw_answer = self.api_provider.api_request(prompt, api=self.api_name)
        llm_answer = raw_answer.strip().replace("\n", " ")

        print('============== Answer ==============')
        print(llm_answer)

        latest_action = "NOOP"
        latest_index = -1
        for action in self.available_actions:
            index = llm_answer.rfind(action)  # Find last occurrence
            if index > latest_index:
                latest_index = index
                latest_action = action

        self.history.append(f"{self.step_counter}: {llm_answer}")
        self.action_history.append(f"{self.step_counter}: {latest_action}")
        self.step_counter += 1

        return self.available_actions.index(latest_action)

    def reset_state(self):
        """Resets the step counter and clears action history."""
        self.step_counter = 0
        self.history.clear()
        self.action_history.clear()


def main():
    text_observation = """
    -5, -4: grass -4, -4: grass -3, -4: grass -2, -4: grass -1, -4: grass 0, -4: grass 1, -4: sand 2, -4: sand 3, -4: sand 4, -4: water 5, -4: water 
    -5, -3: grass -4, -3: grass -3, -3: grass -2, -3: grass -1, -3: grass 0, -3: grass 1, -3: grass 2, -3: sand 3, -3: sand 4, -3: water 5, -3: water 
    -5, -2: grass -4, -2: grass -3, -2: grass -2, -2: grass -1, -2: grass 0, -2: grass 1, -2: grass 2, -2: grass 3, -2: sand 4, -2: water 5, -2: water 
    -5, -1: grass -4, -1: tree -3, -1: grass -2, -1: grass -1, -1: grass 0, -1: grass 1, -1: grass 2, -1: grass 3, -1: grass 4, -1: sand 5, -1: water 
    -5, 0: grass -4, 0: grass -3, 0: grass -2, 0: grass -1, 0: grass 0, 0: grass 1, 0: grass 2, 0: grass 3, 0: grass 4, 0: grass 5, 0: water 
    -5, 1: grass -4, 1: grass -3, 1: grass -2, 1: grass -1, 1: grass 0, 1: grass 1, 1: grass 2, 1: grass 3, 1: grass 4, 1: sand 5, 1: sand 
    -5, 2: grass -4, 2: grass -3, 2: tree -2, 2: grass -1, 2: grass 0, 2: grass 1, 2: grass 2, 2: grass 3, 2: sand 4, 2: sand 5, 2: sand 
    -5, 3: grass -4, 3: stone -3, 3: path -2, 3: iron -1, 3: grass 0, 3: grass 1, 3: grass 2, 3: grass 3, 3: sand 4, 3: sand 5, 3: sand 
    -5, 4: grass -4, 4: stone -3, 4: iron -2, 4: stone -1, 4: coal 0, 4: grass 1, 4: stone 2, 4: stone 3, 4: stone 4, 4: stone 5, 4: stone 
    
    Inventory:
    Wood: 0; Stone: 0; Coal: 0; Iron: 0; Diamond: 0; Sapling: 0; Wood Pickaxe: 0; Stone Pickaxe: 0; Iron Pickaxe: 0; Wood Sword: 0; Stone Sword: 0; Iron Sword: 0; 
    Player Status: Health: 9; Food: 9; Drink: 9; Energy: 9; Sleeping: False; Recover: 3.00; Hunger: 3.00; Thirst: 3.00; Fatigue: 3.00; Direction: 3; 
    Light Level: 0.82; Current Timestep: 3
    """

    agent = ApiAgent(api_name='input', dummy_mode=False)
    instruction = "Set up a furnace for smelting."
    for _ in range(10):
        agent.act(text_observation, instruction)


if __name__ == '__main__':
    main()
