import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import defaultdict
from Util.Config import Config
from Util.util import *

tasks_coords = {
    'T1': (10, 20), 'T2': (30, 50), 'T3': (80, 10),
    'T4': (50, 80), 'T5': (90, 90), 'T6': (20, 70),
    'T7': (70, 40),
    'Start_A': (0, 0),
    'Start_B': (5, 5)
}

# solution_schedule = [
#     {'agent': 'Agent A (Drone)', 'task': 'T1', 'start_time': 4.7, 'end_time': 9.7},
#     {'agent': 'Agent A (Drone)', 'task': 'T6', 'start_time': 12.2, 'end_time': 17.2},
#     {'agent': 'Agent A (Drone)', 'task': 'T4', 'start_time': 18.8, 'end_time': 23.8},
#
#     {'agent': 'Agent B (Robot)', 'task': 'T3', 'start_time': 5.6, 'end_time': 10.6},
#     {'agent': 'Agent B (Robot)', 'task': 'T7', 'start_time': 12.1, 'end_time': 17.1},
#     {'agent': 'Agent B (Robot)', 'task': 'T2', 'start_time': 20.2, 'end_time': 25.2},
# ]


def preprocess_schedule(solution):
    time_line_map = {robot : [] for robot in range(1, Config.ROBOT_NUM + 1)}


    instance = solution.instance
    sequence_map = solution.sequence_map
    info_map = {task: dict() for task in sequence_map.keys()}

    enabled_task_list = []
    task_require_robot_state = [[0 for _ in range(Config.ROBOT_TYPE_NUM)] for i in
                                range(solution.task_num)]

    for path_index, init_task in solution.path_init_task_map.items():
        if init_task != 0:
            robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
            task_require_robot_state[init_task - 1][robot_type - 1] = 1
            info_map[init_task][f'robot_{robot_type}_pre_complete_time'] = 0
            if instance[init_task - 1][5] == task_require_robot_state[init_task - 1]:
                enabled_task_list.append(init_task)

    completed_task_num = 0
    while completed_task_num < solution.task_num:
        idx = random.randrange(len(enabled_task_list))
        enabled_task = enabled_task_list.pop(idx)

        source_position = [instance[enabled_task - 1][1], instance[enabled_task - 1][2]]

        arrival_time_map = {}

        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[enabled_task - 1][5][robot_type - 1] == 0:
                continue
            pre_complete_time = info_map[enabled_task][f'robot_{robot_type}_pre_complete_time']
            robot = sequence_map[enabled_task][f'robot_{robot_type}']
            if sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] == 0:
                pre_position = Config.DEPOT
            else:
                pre_position = [instance[sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][1],
                                instance[sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][2]]

            distance = get_distance(pre_position, source_position)
            travel_time = distance / Config.VELOCITY
            arrival_time = pre_complete_time + travel_time

            arrival_time_map[robot_type] = arrival_time

            time_line_map[robot].append(
                {
                    'task_id': enabled_task,
                    'from_pos': pre_position,
                    'to_pos': source_position,
                    'start_travel_time': pre_complete_time,
                    'arrival_time': arrival_time,
                }
            )

            info_map[enabled_task][f"robot_{robot_type}_arrival_time"] = arrival_time

            next_task = sequence_map[enabled_task][f'robot_{robot_type}_next_task']
            if next_task != 0:
                task_require_robot_state[next_task - 1][robot_type - 1] = 1
                if task_require_robot_state[next_task - 1] == instance[next_task - 1][5]:
                    enabled_task_list.append(next_task)

        execute_time = max(arrival_time_map.values())

        complete_time_list = []
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[enabled_task - 1][5][robot_type - 1] == 0:
                continue
            robot = sequence_map[enabled_task][f'robot_{robot_type}']
            complete_time = execute_time + instance[enabled_task - 1][4][robot_type - 1]

            time_line_map[robot][-1]['departure_time'] = complete_time
            next_task = sequence_map[enabled_task][f'robot_{robot_type}_next_task']
            if next_task != 0:
                info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time

            complete_time_list.append(complete_time)

        final_complete_time = max(complete_time_list)

        completed_task_num += 1

    return time_line_map


def plot(solution):
    agent_timeline = preprocess_schedule(solution)
    fig, ax = plt.subplots(figsize=(10, 8))
    agents = list(agent_timeline.keys())

    cmap = plt.colormaps.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(agents)))
    agent_color_map = {agent: colors[i] for i, agent in enumerate(agents)}
    instance = solution.instance
    def update(frame):
        ax.clear()
        current_time = frame

        for task_id, task_info in enumerate(instance):
            ax.scatter(task_info[1], task_info[2], s=100, c='gray', marker='s')
            ax.text(task_info[1], task_info[2] + 0.03, task_id + 1, ha='center', va='bottom', fontsize=9)

        # for task_id, pos in tasks_coords.items():
        #         ax.scatter(pos[0], pos[1], s=100, c='gray', marker='s')
        #         ax.text(pos[0], pos[1] + 3, task_id, ha='center', va='bottom', fontsize=9)

        for agent, timeline in agent_timeline.items():
            if not timeline: continue

            agent_color = agent_color_map[agent]

            for step in timeline:
                if current_time >= step['arrival_time']:
                    ax.plot([step['from_pos'][0], step['to_pos'][0]],
                            [step['from_pos'][1], step['to_pos'][1]],
                            color=agent_color, linestyle='-', linewidth=2, alpha=0.7)

            agent_current_pos = None
            for step in reversed(timeline):
                if step['start_travel_time'] < current_time < step['arrival_time']:
                    progress = (current_time - step['start_travel_time']) / (
                            step['arrival_time'] - step['start_travel_time'])
                    x = step['from_pos'][0] + progress * (step['to_pos'][0] - step['from_pos'][0])
                    y = step['from_pos'][1] + progress * (step['to_pos'][1] - step['from_pos'][1])
                    agent_current_pos = (x, y)
                    ax.plot([step['from_pos'][0], x], [step['from_pos'][1], y], color=agent_color, linestyle='--',
                            linewidth=2)
                    break
                elif step['arrival_time'] <= current_time <= step['departure_time']:
                    agent_current_pos = step['to_pos']
                    break

            if agent_current_pos is None:
                if current_time <= timeline[0]['start_travel_time']:
                    agent_current_pos = timeline[0]['from_pos']
                else:
                    agent_current_pos = timeline[-1]['to_pos']

            ax.scatter(agent_current_pos[0], agent_current_pos[1], s=150, c=[agent_color], marker='*',
                       edgecolors='black',
                       zorder=10)
            ax.text(agent_current_pos[0], agent_current_pos[1] - 5, agent.split(' ')[0], ha='center', va='top',
                    fontsize=10,
                    color=agent_color, weight='bold')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Parallel Agent Trajectory | Time: {current_time:.2f} s")
        ax.grid(True, linestyle=':')
        ax.set_aspect('equal', adjustable='box')

        legend_handles = [plt.Line2D([0], [0], color=agent_color_map[agent], lw=4, label=agent) for agent in agents]
        ax.legend(handles=legend_handles, loc='upper left')

    max_time = 0
    for timeline in agent_timeline.values():
        if timeline:
            max_time = max(max_time, timeline[-1]['departure_time'])

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, max_time + 1, 0.1), interval=50, repeat=False)

    ani.save('parallel_agent_path_animation.gif', writer='pillow', fps=20)

    plt.show()

# agent_timeline = preprocess_schedule(solution_schedule, tasks_coords)






#
# max_time = 0
# for timeline in agent_timeline.values():
#     if timeline:
#         max_time = max(max_time, timeline[-1]['departure_time'])
#
# ani = animation.FuncAnimation(fig, update, frames=np.arange(0, max_time + 1, 0.1), interval=50, repeat=False)
#
# ani.save('parallel_agent_path_animation.gif', writer='pillow', fps=20)
#
# plt.show()


