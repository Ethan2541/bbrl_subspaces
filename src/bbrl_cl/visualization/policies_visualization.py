import os
import re
import shutil

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


import numpy as np
import gym
from datetime import datetime




from bbrl_algos.models.loggers import Logger
from bbrl.agents import Agent, TemporalAgent, Agents
from bbrl_algos.algos.dqn.dqn import local_get_env_agents
from bbrl_algos.models.critics import DiscreteQAgent
from bbrl_algos.models.actors import ContinuousDeterministicActor
from bbrl_algos.models.critics import ContinuousQAgent


from bbrl.workspace import Workspace
from bbrl_algos.models.envs import get_eval_env_agent

import plotly.graph_objects as go
from cvxopt import matrix, solvers
from matplotlib.colors import LinearSegmentedColormap

from visualization_tools import save




# def extract_number_from_filename(filename):
#     """
#     Extracts the numerical value (XXX) from a filename of the form "CartPole-v1dqnXXX.agt"True
#     """
#     match = re.search(r'CartPole-v1dqn(\d+\.\d+).agt', filename)
#     if match:
#         return float(match.group(1))
#     return None

def extract_number_from_filename(filename, env_name_algo):
    """
    Extracts the numerical value (XXX) from a filename of the form "{env_name_algo}XXX.agt"
    """
    pattern = rf'{env_name_algo}(\d+\.\d+).agt'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    return None
    

def get_last_chronological_folder(source_dir):
    """
    Returns the filename of the last chronological folder by time within the specified source directory
    """
    # Get a list of folders in chronological order by time
    folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))],
                     key=lambda x: os.path.getmtime(os.path.join(source_dir, x)),
                     reverse=True)
    
    # Return the filename of the first folder (last chronological folder)
    return folders[0] if folders else None



def extract_number_from_filename_traj(filename):
    # Extract the number after the underscore
    parts = filename.split('_')
    if len(parts) > 1:
        try:
            return int(parts[-1].split('.')[0])
        except ValueError:
            pass
    return float('inf')


def read_and_sort_agents(dirname, env_name): #dirname, env_name
        script_directory = os.path.dirname(os.path.abspath(__file__))
        #source_dir = os.path.join(script_directory, "dqn_agents", "CartPole-v1")
        source_dir = os.path.join(script_directory, dirname, env_name)
        #print(source_dir)
        
        # Get a list of folders in chronological order
        agt_files = [f for f in os.listdir(source_dir) if f.endswith(".agt")]
        agt_files.sort(key=extract_number_from_filename_traj)
        loaded_agents = []

        agent = Agent()

        #print(agt_files)
    
        # Iterate over each file and load the agent using agent.load_model
        for file in agt_files:
            file_path = os.path.join(source_dir, file)  # Full path to the agent file
            loaded_agent = agent.load_model(file_path)
            loaded_agents.append(loaded_agent)
        
        return loaded_agents


def get_best_policy(date, time, suffix, env_name_algo, algo, directory):
    """
    Stocker la meilleure politique dans un fichier
    """
    # Path to the directory containing the best agents
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_directory, "..", "algos", algo, "tmp", "hydra", date, time, directory)
    
    # Path to the destination directory in the visualization folder
    destination_dir = os.path.join(script_directory, directory)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Get a list of folders in chronological order
    agt_files = [f for f in os.listdir(source_dir) if f.endswith(".agt")]
    
    # Extract the values from each folder and append them to an array
    env_algo = env_name_algo + algo
    values_array = [extract_number_from_filename(agt_file, env_algo) for agt_file in agt_files]
    
    filtered_values = [val for val in values_array if val is not None]
    #print(filtered_values)

    # Trouver l'indice du maximum dans filtered_values
    if filtered_values:
        max_index = values_array.index(max(filtered_values))
    else:
        # Gérer le cas où il n'y a aucun élément valide dans values_array
        max_index = None
    
    # Get the folder with the maximum value
    max_value_agt_file = agt_files[max_index]
    
    # Copy the files from the folder with the max value to the destination directory
    source_file = os.path.join(source_dir, max_value_agt_file)
    
    # Modify the destination file name to include the suffix
    base_name, extension = os.path.splitext(max_value_agt_file)
    destination_file = os.path.join(destination_dir, f"{base_name}_{suffix}{extension}")
    
    shutil.copyfile(source_file, destination_file)
    
    # Print the results
    # print("Values array:", values_array)
    # print("File with max value:", max_value_agt_file)
    # print("File with max value copied successfully to:", destination_dir)


def load_best_agent(directory):
    """
    Load all the best agents in the dqn_best_agents folder using agent.load_model and store them in a list
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_directory, directory)

    # Get a list of files in the dqn_best_agents folder
    agt_files = [f for f in os.listdir(source_dir) if f.endswith(".agt")]

    # List to store the loaded agents
    loaded_agents = []

    agent = Agent()
    
    # Iterate over each file and load the agent using agent.load_model
    for file in agt_files:
        file_path = os.path.join(source_dir, file)  # Full path to the agent file
        loaded_agent = agent.load_model(file_path)
        loaded_agents.append(loaded_agent)
    
    return loaded_agents


def load_policies(loaded_agents):
    list_policies = []
    for i in range (len(loaded_agents)):
        policy = torch.nn.utils.parameters_to_vector(loaded_agents[i].parameters())
        list_policies.append(policy)
    return list_policies


def plot_triangle_with_multiple_points(alpha_reward_list, alpha_reward_traj, plot_traj):
    """
    Plot a triangle with vertices representing policies and a new point calculated as a weighted sum of policies.
    Color the new point based on its reward value.

    Parameters:
    - coefficients_list (list of lists): List of coefficients (a1, a2, a3) used to calculate the new point.
    - rewards_list (list of float): List of reward values for each point.
    """

    # Generate the vertices of the equilateral triangle
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2],[0,0]])

    # Plot the triangle edges
    plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-')

    # Plot policy vertices
    plt.text(triangle_vertices[0, 0] - 0.05, triangle_vertices[0, 1] - 0.03, 'p1', fontsize=10)
    plt.text(triangle_vertices[1, 0] + 0.05, triangle_vertices[1, 1] - 0.03, 'p2', fontsize=10)
    plt.text(triangle_vertices[2, 0], triangle_vertices[2, 1] + 0.03, 'p3', fontsize=10)

    #norm = mcolors.Normalize(vmin=0, vmax=500) #TO TO HAVE THE VALUES FROM 0 TO 500
    alphas_list, rewards_list = zip(*alpha_reward_list)
    # Normalize rewards for colormap
    norm = plt.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))
    norm = mcolors.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))

    # Choose a colormap that covers the entire range of rewards
    cmap = plt.get_cmap('RdBu_r')
    #cmap = plt.get_cmap('viridis')

    # Plot the points with adjusted positions and transparency
    for coefs, reward in (alpha_reward_list):
        #print(coefs, reward)
        new_point = np.dot(np.array(coefs), triangle_vertices[:3])
        #jitter = np.random.normal(0, 0.01, 2)  # Add a small random jitter to avoid superposition
        #plt.scatter(new_point[0] + jitter[0], new_point[1] + jitter[1], c=reward, cmap=cmap, norm=norm, alpha=0.5, s=250)
        plt.scatter(new_point[0], new_point[1], c=reward, cmap=cmap, norm=norm, s=60)
    # Set axis limits and labels
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)

    if(plot_traj):
        for coefs, reward in (alpha_reward_traj):
            new_point = np.dot(np.array(coefs), triangle_vertices[:3])
            plt.scatter(new_point[0], new_point[1], c=reward, cmap=cmap, norm=norm, s=60)
        

        # Define colors
        yellow = [1, 1, 0]  # RGB values for yellow
        green = [0, 0.5, 0]   # RGB values for green
        power = 1
        # Plot trajectory with gradient color
        for i in range(1, len(alpha_reward_traj)):
            coefs1, _ = alpha_reward_traj[i-1]
            coefs2, _ = alpha_reward_traj[i]
            interpolation = i / len(alpha_reward_traj)
            color = [yellow[j] + (green[j] - yellow[j]) * (interpolation ** power) for j in range(3)]
            plt.plot([np.dot(np.array(coefs1), triangle_vertices[:3])[0], np.dot(np.array(coefs2), triangle_vertices[:3])[0]], 
                    [np.dot(np.array(coefs1), triangle_vertices[:3])[1], np.dot(np.array(coefs2), triangle_vertices[:3])[1]], 
                    color=color , linewidth=2.5)

    # Add color bar legend
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca())
    cbar.set_label('Reward')


    # Add legend

    # Define custom legend elements
    
    if(plot_traj):
        legend_elements = [
            mlines.Line2D([], [], color='yellow', marker='o', markersize=10, label='Source'),
            mlines.Line2D([], [], color='green', marker='o', markersize=10, label='Destination'),
            mlines.Line2D([], [], color='yellow', linewidth=1, label='Policies Trajectory'),
            mlines.Line2D([], [], color='black', linewidth=1, label='Triangle Edges'),
        ]
        
    else:
        legend_elements = [
            mlines.Line2D([], [], color='black', linewidth=1, label='Triangle Edges')
        ]


    # Add legend
    plt.legend(handles=legend_elements)


    script_directory = os.path.dirname(os.path.abspath(__file__))


    save_directory = os.path.join(script_directory, "..", "..","..", "visualization_results")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Then save the plot using the save_directory
    save_path = os.path.join(save_directory, f"policy_{date_time}.png")

    plt.savefig(save_path)

    # Show the plot
    plt.show()



def plot_triangle_with_multiple_points_plotly(alpha_reward_list, alpha_reward_traj, plot_traj):
    """
    Plot a triangle with vertices representing policies and a new point calculated as a weighted sum of policies.
    Color the new point based on its reward value.

    Parameters:
    - coefficients_list (list of lists): List of coefficients (a1, a2, a3) used to calculate the new point.
    - rewards_list (list of torch.Tensor): List of reward tensors for each point.
    """

    # Generate the vertices of the equilateral triangle
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0,0]])

    # Create trace for triangle edges
    triangle_edges = go.Scatter(
        x=triangle_vertices[:, 0],
        y=triangle_vertices[:, 1],
        mode='lines',
        line=dict(color='black'),
        showlegend=False,  # Set this to False to hide from legend
    )

    # Create trace for policy vertices
    policy_vertices = go.Scatter(
        x=[triangle_vertices[0, 0], triangle_vertices[1, 0], triangle_vertices[2, 0]],
        y=[triangle_vertices[0, 1], triangle_vertices[1, 1], triangle_vertices[2, 1]],
        mode='text',
        text=['P1', 'P2', 'P3'],
        textposition='top center',
        textfont=dict(size=20),
        showlegend=False,  # Set this to False to hide from legend
    )

    # Create trace for new points
    alphas_list, rewards_list = zip(*alpha_reward_list)
    # Convert Torch tensors to standard Python numbers
    rewards_list = [reward.item() for reward in rewards_list]
    #print(rewards_list)

    # Normalize rewards for colormap
    norm = plt.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))
    min_reward = min(rewards_list)
    max_reward = max(rewards_list)


    # Create trace for new points with the updated colorbar attributes
    new_points = []
    for coefs, reward in (alpha_reward_list):
        reward = reward.item()
        new_point = np.dot(np.array(coefs), triangle_vertices[:3])
        hover_text = f"Reward: {reward:.2f}<br>Coordinates: ({new_point[0]:.2f}, {new_point[1]:.2f})<br>Alphas: {coefs}"
        new_points.append(go.Scatter(
            x=[float(new_point[0])],
            y=[float(new_point[1])],
            mode='markers',
            marker=dict(
                color=[reward],  # Enclose reward in a list to define its color based on the 'RdBu' scale
                size=10,
                colorbar=dict(
                    title='Reward',
                    tickvals=[min_reward, max_reward],
                    ticktext=[f'{min_reward:.2f}', f'{max_reward:.2f}']
                ),
                colorscale='RdBu_r',  # Use the built-in Red-Blue color scale
                cmin=min_reward,  # Explicitly set the min for color scaling
                cmax=max_reward,  # Explicitly set the max for color scaling
                showscale=True  # Ensure that the colorscale is shown
            ),
            text=[hover_text],
            hoverinfo='text',
            showlegend=False,
        ))


    new_points_traj = []

    for coefs, reward in alpha_reward_traj:
        reward = reward.item()
        new_point = np.dot(np.array(coefs), triangle_vertices[:3])
        hover_text = f"Reward: {reward:.2f}<br>Coordinates: ({new_point[0]:.2f}, {new_point[1]:.2f})<br>Alphas: {coefs}"
        reward_color = mcolors.rgb2hex(plt.cm.RdBu_r(norm(reward)))
        new_points_traj.append(go.Scatter(
            x=[float(new_point[0])],
            y=[float(new_point[1])],
            mode='markers',
            marker=dict(
                color=reward_color,
                size=10,
                colorbar=dict(
                    title='Reward',
                    tickvals=[min_reward, max_reward],
                    ticktext=[f'{min_reward:.2f}', f'{max_reward:.2f}']
                ),
                colorscale='RdBu_r',
                cmin=min_reward,
                cmax=max_reward,
                showscale=True
            ),
            text=[hover_text],
            hoverinfo='text',
            showlegend=False,
        ))

    if(plot_traj):
        # Convert alpha_reward_traj to Plotly traces for trajectory
        # Define the custom color scale from yellow to green
        colormap = plt.cm.summer_r  # Choose a colormap (you can change it if needed)
        num_segments = len(alpha_reward_traj) - 1
        traj_lines = []
        for i in range(num_segments):
            coefs1, _ = alpha_reward_traj[i]
            coefs2, _ = alpha_reward_traj[i + 1]
            interpolation = i / num_segments
            color = mcolors.rgb2hex(colormap(interpolation))
            traj_lines.append(go.Scatter(
                x=[np.dot(np.array(coefs1), triangle_vertices[:3])[0], np.dot(np.array(coefs2), triangle_vertices[:3])[0]],
                y=[np.dot(np.array(coefs1), triangle_vertices[:3])[1], np.dot(np.array(coefs2), triangle_vertices[:3])[1]],
                mode='lines',
                line=dict(color=color, width=2.5),
                showlegend=False,
            ))


    # Create layout
    layout = go.Layout(
        title='Triangle Plot',
        xaxis=dict(title='X-axis', range=[-0.1, 1.1]),
        yaxis=dict(title='Y-axis', range=[-0.1, np.sqrt(3)/2 + 0.1]),
        showlegend=True
    )

    # Plot
    if(plot_traj):
        fig = go.Figure(data=[triangle_edges, policy_vertices] + new_points + new_points_traj + traj_lines, layout=layout)
        fig.show()
    else:
        fig = go.Figure(data=[triangle_edges, policy_vertices] + new_points, layout=layout)
        fig.show()


def update_policy_with_coefficients(list_policies, coefficients):
    """
    Returns policy = a1 * p1 + a2 * p2 + a3 * p3
    """

    # Ensure the number of coefficients matches the number of policy tensors
    if len(list_policies) != len(coefficients):
        raise ValueError("Number of policy tensors and coefficients must match.")

    # Initialize the updated policy tensor with zeros
    updated_policy = torch.zeros_like(list_policies[0])

    # Update the policy tensor using the provided coefficients
    for coeff, policy in zip(coefficients, list_policies):
        updated_policy += coeff * policy

    return updated_policy

def create_new_DQN_agent(cfg, env_agent):

    # Create the agent
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    policy = globals()["DiscreteQAgent"](
        obs_size, cfg.algorithm.architecture.hidden_sizes, act_size
    )
    ev_agent = Agents(env_agent, policy)
    eval_agent = TemporalAgent(ev_agent)
    return eval_agent

def create_new_TD3_agent(cfg, env_agent):
    # Create the agent
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    policy = globals()["ContinuousDeterministicActor"](
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    ev_agent = Agents(env_agent, policy)
    eval_agent = TemporalAgent(ev_agent)
    return eval_agent

def evaluate_agent(eval_agent, theta):

    # Calculate the reward
    torch.nn.utils.vector_to_parameters(theta, eval_agent.parameters())

    workspace = Workspace()

    eval_agent(workspace, t=0, stop_variable="env/done", render=False)

    rewards = workspace["env/cumulated_reward"][-1]
    #print(rewards)
    mean_reward = rewards.mean()

    return mean_reward


def intersection_point(m, b, x_coord):
    "Calculates the intersection point of an y = mx'+b axis with the x = x_coord axis"
    # Calculate y-coordinate of intersection point
    y = m * (x_coord) + b
    return x_coord, y

def find_axis_through_point(point, slope):
    # Point coordinates
    x_h, y_h = point
    
    # Equation of the axis passing through H: y = mx + b
    # We know that the slope is the same as the given slope
    # So, b = y - mx
    b = y_h - slope * x_h
    
    # Return the equation of the axis: y = slope * x + b
    return slope, b

def generate_left_edge_points(num_points):
    # Generate x values linearly spaced between 0 and 1/2
    x_values = np.linspace(0, 0.5, num_points)
    
    # Calculate corresponding y values based on the equation sqrt(3) * x
    y_values = np.sqrt(3) * x_values
    
    # Combine x and y values into a list of points
    points = list(zip(x_values, y_values))
    
    return points

def generate_lower_edge_points(num_points):
    # Generate x values linearly spaced between 0 and 1
    x_values = np.linspace(0, 1, num_points)
    
    # Set y values to zero for all points
    y_values = np.zeros_like(x_values)
    
    # Combine x and y values into a list of points
    points = list(zip(x_values, y_values))
    
    return points

def get_alphas_from_point(x,y):
    triangle_vertices = [[0, 0],[1, 0], [0.5, np.sqrt(3)/2]]
    #print("triangle_vertices : ", triangle_vertices)
    A = np.vstack([triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]]).T
    #print("A : ", A)

    A = np.vstack([A, np.ones(3)])

    b = np.array([x, y, 1])  

    # Résolution du système d'équations
    alphas = np.linalg.lstsq(A, b, rcond=None)[0]

    # Normalisation pour s'assurer que les alphas sont positifs ou nuls et somment à 1
    alphas = np.maximum(alphas, 0)  # Les valeurs négatives deviennent 0
    alphas /= np.sum(alphas)  # Normalisation pour que la somme soit égale à 1

    #print("alphas: ",alphas)

    return alphas

def is_inside_triangle(point, A, B, C):

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(point, A, B) < 0.0
    b2 = sign(point, B, C) < 0.0
    b3 = sign(point, C, A) < 0.0

    return ((b1 == b2) and (b2 == b3))


def policies_visualization(eval_agent, num_points, loaded_policies, policies_traj, plot_traj):

    #1. Calculating alphas and rewards

    points = generate_left_edge_points(num_points)
    points2 = generate_lower_edge_points(num_points)

    axis_equations_points = []
    i=0
    slope = -np.sqrt(3)/3
    for point in points:
        new_slope, b = find_axis_through_point(point,slope)
        axis_equations_points.append((new_slope, b))

    intersection = []
    i = 0
    j = 0
    for i in range(len(axis_equations_points)):
        for j in range(len(points2)):
            slope, b = axis_equations_points[i]
            x_coord = points2[j][0]
            intersection_point_val = intersection_point(slope, b, x_coord)
            # Check if the intersection point is not already in points or points2
            if (intersection_point_val not in points) and (intersection_point_val not in points2):
                # Check if the intersection point lies in the positive quadrant
                if intersection_point_val[0] > 0 and intersection_point_val[1] > 0:
                    intersection.append(intersection_point_val)

    #print(intersection)
    #print(intersection)

    # Extract x and y coordinates from points
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]

    # Extract x and y coordinates from points2
    x_points2 = [point[0] for point in points2]
    y_points2 = [point[1] for point in points2]

    # Extract x and y coordinates from intersection
    x_intersection = [point[0] for point in intersection]
    y_intersection = [point[1] for point in intersection]

    alpha_reward_list = []  # List to store tuples of alphas and rewards
    cpt=0

    # Calculate the particular point reward (tip of the triangle)
    theta = update_policy_with_coefficients(loaded_policies, [0, 0, 1])
    reward = evaluate_agent(eval_agent, theta)
    alpha_reward_list.append(([0, 0, 1], reward))

    for i in range(len(x_points)):
        if is_inside_triangle([x_points[i], y_points[i]], [0, 0], [1, 0], [0.5, np.sqrt(3) / 2]):
            if(x_points[i] != 0.5 and y_points[i] != np.sqrt(3)/2):
                #print(x_points[i], y_points[i])
                alpha = get_alphas_from_point(x_points[i], y_points[i])
                #print(alpha)
                theta = update_policy_with_coefficients(loaded_policies, alpha)
                #print("theta : ", theta)
                #print("theta : ", theta.min(), theta.max(), theta.mean())
                #array = theta.detach().numpy()
                #print(array)
                reward = evaluate_agent(eval_agent, theta)
                #print(reward.item())
                alpha_reward_list.append((alpha, reward))
                cpt+=1
                print(cpt)

    for j in range(len(x_points2)):
        if is_inside_triangle([x_points2[j], y_points2[j]], [0, 0], [1, 0], [0.5, np.sqrt(3) / 2]):
            if(x_points2[i] != 0.5 and y_points2[i] != np.sqrt(3)/2):
                alpha = get_alphas_from_point(x_points2[j], y_points2[j])
                theta = update_policy_with_coefficients(loaded_policies, alpha)
                #print(evaluate_agent(eval_agent, theta))
                reward = evaluate_agent(eval_agent, theta)
                alpha_reward_list.append((alpha, reward))
                cpt+=1
                print(cpt)

    for k in range(len(x_intersection)):
        if is_inside_triangle([x_intersection[k], y_intersection[k]], [0, 0], [1, 0], [0.5, np.sqrt(3) / 2]):
            if(x_intersection[i] != 0.5 and y_intersection[i] != np.sqrt(3)/2):
                #print(x_intersection[k], y_intersection[k])
                alpha = get_alphas_from_point(x_intersection[k], y_intersection[k])
                #print(alpha)
                theta = update_policy_with_coefficients(loaded_policies, alpha)
                reward = evaluate_agent(eval_agent, theta)
                #print(reward.item())
                alpha_reward_list.append((alpha, reward))
                cpt+=1
                print(cpt)

    alpha_reward_traj = []
    p1 = loaded_policies[0].detach().numpy()
    p2 = loaded_policies[1].detach().numpy()
    p3 = loaded_policies[2].detach().numpy()

    # print("p1 : ", p1)
    # print("p2 : ", p2)
    # print("p3 : ", p3)

    if(plot_traj):
        for policy in policies_traj:
            policy = policy.detach().numpy()
            a1, a2, a3 = projection_convex_hull(policy, p1, p2, p3)
            coeff_list = [a1.item(), a2.item(), a3.item()]  # Convert NumPy arrays to Python scalars
            p_prime = update_policy_with_coefficients(loaded_policies, coeff_list)
            #print(p_prime)
            reward_traj = evaluate_agent(eval_agent, p_prime)
            #print(reward_traj.item())
            alpha_reward_traj.append((coeff_list, reward_traj))

    #3. Plotting the triangle with the points
    plot_triangle_with_multiple_points_plotly(alpha_reward_list, alpha_reward_traj, plot_traj)
    plot_triangle_with_multiple_points(alpha_reward_list, alpha_reward_traj, plot_traj)


def projection_convex_hull(p, p1, p2, p3):
    """
    Find the projection of a point onto the convex hull defined by three points p1, p2 and p3.

    Parameters:
        p (numpy.array): The point to be projected.
        p1 (numpy.array): First vector defining the convex hull.
        p2 (numpy.array): Second vector defining the convex hull.
        p3 (numpy.array): Third vector defining the convex hull.

    Returns:
        numpy.array: The projected points coefficients.
    """
    # Ensure all vectors have the same dimension
    assert p.shape == p1.shape == p2.shape == p3.shape, "All vectors must have the same dimension"

    n = 3
    # Construct the P and q matrices for the QP problem
    p_mat = 2 * np.array(
        [[np.dot(p1, p1.T), np.dot(p1, p2.T), np.dot(p1, p3.T)],
         [np.dot(p1, p2.T), np.dot(p2, p2.T), np.dot(p2, p3.T)],
         [np.dot(p1, p3.T), np.dot(p2, p3.T), np.dot(p3, p3.T)]])
    q_mat = -2 * np.array([np.dot(p, p1),
                           np.dot(p, p2),
                           np.dot(p, p3)])

    g_mat = np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
    h_mat = np.array([0, 0, 0])
    a_mat = np.array([[1, 1, 1]])
    b_mat = np.array([1.])

    p_mat = p_mat.astype(np.double)
    q_mat = q_mat.astype(np.double)
    g_mat = g_mat.astype(np.double) 
    h_mat = h_mat.astype(np.double)
    a_mat = a_mat.astype(np.double)
    b_mat = b_mat.astype(np.double)

    # Convert matrices to cvxopt format
    p_mat = matrix(p_mat, (n, n), 'd')
    q_mat = matrix(q_mat, (n, 1), 'd')
    g_mat = matrix(g_mat, (n, n), 'd')
    h_mat = matrix(h_mat, (n, 1), 'd')
    a_mat = matrix(a_mat, (1, n), 'd')
    b_mat = matrix(b_mat, (1, 1), 'd')

    sol = solvers.qp(p_mat, q_mat, G=g_mat, h=h_mat, A=a_mat, b=b_mat)

    # Convert solution to triple (x, y, z)
    a1, a2, a3 = np.array(sol['x'])

    # Return projected point
    return a1, a2, a3



def evo_algo(nbeval, nb_individuals, triangle_policies, agent, noise_scale=0.1):
    bestFit = float('-inf')
    bestIt = 0
    dimensions = len(triangle_policies)
    traj = []

    for nb_iterations in range(nbeval):
        
        # Generate nb_individuals random alphas whose sum is equal to 1
        alphas = np.random.uniform(0, 1, (nb_individuals, dimensions))

        # Normalize alphas so that their sum equals 1
        alphas /= np.sum(alphas, axis=1)[:, np.newaxis]

        # Compute the new policies
        thetas = [update_policy_with_coefficients(triangle_policies, alpha) for alpha in alphas]

        # Adding noise to the policies
        noisy_thetas = [theta + noise_scale * torch.randn(len(theta)) for theta in thetas]
        
        # Compute the new rewards adding some noise
        rewards = [evaluate_agent(agent, noisy_theta).item() for noisy_theta in noisy_thetas]

        sorted_indices = np.argsort(rewards)

        # Take the 3 best-performing policies
        best_3_policies = [thetas[idx] for idx in sorted_indices[:3]]

        # Store the best policy
        traj.append(best_3_policies[0])
        print(traj)

        sorted_indices = sorted_indices[::-1]

        if rewards[sorted_indices[0]] > bestFit:
            bestFit = rewards[sorted_indices[0]]
            bestIt = nb_iterations
        
        # Update the policies used at each iteration
        triangle_policies = best_3_policies
        print("Iteration", nb_iterations)
        print("Best fit", bestFit, "at iteration", bestIt)

    print("Best fit", bestFit, "at iteration", bestIt)

    return bestFit, traj



@hydra.main(
    config_path="../algos/dqn/configs/",
    config_name="dqn_cartpole.yaml",
    #config_path="../algos/td3/configs/",
    #config_name="td3_swimmer.yaml",
)  # , version_base="1.3")

def main(cfg_raw: DictConfig):
    #print(read_tensor_file())

    date = "2024-03-20" 
    time = "12-11-42"
    get_best_policy(date, time, 1, "CartPole-v1", "dqn", "dqn_best_agents")
                                                                                         
    date = "2024-03-20" 
    time = "12-14-28"
    get_best_policy(date, time, 2, "CartPole-v1", "dqn", "dqn_best_agents")

    date = "2024-03-20" 
    time = "12-16-48"
    get_best_policy(date, time, 3, "CartPole-v1", "dqn", "dqn_best_agents")

    # date = "2024-03-30" 
    # time = "12-48-26"
    # get_best_policy(date, time, 1, "Swimmer-v3", "td3", "td3_best_agents")
                                                                                         
    # time = "12-56-03"
    # get_best_policy(date, time, 2, "Swimmer-v3", "td3", "td3_best_agents")

    # time = "13-03-51"
    # get_best_policy(date, time, 3, "Swimmer-v3", "td3", "td3_best_agents")

    loaded_agents = load_best_agent("dqn_best_agents")
    #loaded_agents = load_best_agent("td3_best_agents_swimmer")     

    _, eval_env_agent = local_get_env_agents(cfg_raw)
    eval_agent = create_new_DQN_agent(cfg_raw, eval_env_agent)
    #eval_agent = create_new_TD3_agent(cfg_raw, eval_env_agent)

    list_agents_traj = read_and_sort_agents("dqn_agents", "CartPole-v1")
    #list_agents_traj = read_and_sort_agents("td3_agents", "Swimmer-v3")
    list_policies_traj = load_policies(list_agents_traj)

    #policies_visualization(eval_agent, 80, load_policies(loaded_agents), list_policies_traj, plot_traj=True)

    # Evolutionary algorithm
    nbeval = 10
    nb_individuals = 20
    triangle_policies = load_policies(loaded_agents)
    best_3_policies, traj = evo_algo(nbeval, nb_individuals, triangle_policies, eval_agent)
    print("Evo algo: ", best_3_policies)

    policies_visualization(eval_agent, 60, load_policies(loaded_agents), traj, plot_traj=True)


if __name__ == "__main__":
    main()

