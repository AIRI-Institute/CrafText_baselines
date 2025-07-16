___

# Interaction step: {{current_step}}

You are an advanced AI agent in the game Crafter. Your goal is to analyze the observation, inventory, and player status,
then generate the best sequence of actions to accomplish the given objective efficiently.

The observation system is centered around the agent, positioned at coordinates (0, 0). 
All other cells are described relative to the agent's position. 
To interact with an object, the agent must approach it by moving to the nearest adjacent cell, align itself towards the object, and perform the corresponding action.

## Observation:
{{ text_observation }}
The coordinates of the agent is always 0, 0, and the world is centered around him.
All other cells are described relative to the agent's position. 
To interact with an object, the agent must approach it by moving to the nearest adjacent cell, 
align itself towards the object, and perform the corresponding action.

## Task Instruction:
{{ instruction }}   

## Actions:
{{ available_actions | join(", ") }}

## Action History:
{{ action_history }}

## Answer History:
{{ history }}

## Expected Response:
Select one appropriate action from the available options based on the input. 
The most recent action provided in your response will be applied to the environment. 

## Next Action I Will do
LEFT

## Next Action I Will do
RIGHT

## Next Action I Will do
RIGHT

## Next Action I Will do
DO

## Next Action I Will do
