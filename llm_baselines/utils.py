import os
import jax.numpy as jnp
from craftax.craftax.constants import MAX_OBS_DIM
from craftax.craftax_classic.constants import OBS_DIM, BlockType
from tabulate import tabulate

from baselines.api_models.api_agent import ApiAgent

import jax

# from craftax.craftax.constants import *
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.envs.craftax_state import Inventory

from craftext.craftext_wrapper import InstructionWrapper

def render_craftax_text(state) -> str:
    text_obs = ""
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Pad map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    def block_name(val):
        try:
            return BlockType(int(val)).name.lower()
        except Exception:
            return "unknown"

    # Mobs
    mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.uint8)  # 4 types of mobs

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_type_index = carry

        local_position = (
            mobs.position[mob_index]
            - state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all()
        on_screen *= mobs.mask[mob_index]

        mob_map = mob_map.at[local_position[0], local_position[1], mob_type_index].set(
            on_screen.astype(jnp.uint8)
        )

        return (mob_map, mobs, mob_type_index), None

    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.zombies, 0),
        jnp.arange(state.zombies.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map, (mob_map, state.cows, 1), jnp.arange(state.cows.mask.shape[0])
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.skeletons, 2),
        jnp.arange(state.skeletons.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.arrows, 3),
        jnp.arange(state.arrows.mask.shape[0]),
    )
 
    def mob_id_to_name(id):
        if id == 0:
            return "Zombie"
        elif id == 1:
            return "Cow"
        elif id == 2:
            return "Skeletons"
        elif id == 3:
            return "Arrows"
       


    table = []
    for x in range(OBS_DIM[0]):
        row = []
        for y in range(OBS_DIM[1]):
            tx, ty = x - OBS_DIM[0] // 2, y - OBS_DIM[1] // 2
            if mob_map[x, y].max() > 0.5:
                mob = f" mob {mob_id_to_name(mob_map[x, y].argmax())} on {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}"
                row.append(mob)
            if tx == 0 and ty == 0:
                row.append(f"agent {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}")
            else:
                row.append(f"{block_name(map_view[x, y])} {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}")

        table.append(row)

    text_obs += "\n" + tabulate(table, tablefmt="github") + "\n"


    # Inventory
    inv = state.inventory
    text_obs += "\nInventory:\n"
    for field in Inventory.__dataclass_fields__:
        val = getattr(inv, field)
        text_obs += f"{field.replace('_', ' ').title()}: {val}; "

    # Player status
    text_obs += "\nPlayer Status: "
    text_obs += f"Health: {state.player_health}; "
    text_obs += f"Food: {state.player_food}; "
    text_obs += f"Drink: {state.player_drink}; "
    text_obs += f"Energy: {state.player_energy}; "
    text_obs += f"Sleeping: {state.is_sleeping}; "
    text_obs += f"Recover: {state.player_recover:.2f}; "
    text_obs += f"Hunger: {state.player_hunger:.2f}; "
    text_obs += f"Thirst: {state.player_thirst:.2f}; "
    text_obs += f"Fatigue: {state.player_fatigue:.2f}; "
    text_obs += f"Direction: {state.player_direction}; "

    # Light level
    text_obs += f"\nLight Level: {state.light_level:.2f}; "

    return text_obs
