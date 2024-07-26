# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go

from datetime import datetime

from .utils import evaluate_agent, find_axis_through_point, generate_left_edge_points, generate_lower_edge_points, get_alphas_from_point, get_point_from_alphas, intersection_point, is_inside_triangle



class SubspaceVisualizer:
    def __init__(self, algorithm_name, env_name, num_points, interactive, thresholds, output_path="./figures/subspace_visualizations", **kwargs):
        self.algorithm_name = algorithm_name
        self.env_name = env_name
        self.is_interactive = interactive
        self.num_points = num_points
        self.thresholds = thresholds
        self.current_thresholds = list(self.thresholds)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path

    
    def reset(self, **kwargs):
        self.current_thresholds = list(self.thresholds)


    def plot_subspace(self, eval_agent, logger, info={}, n_steps=None, **kwargs):
        logger = logger.get_logger(type(self).__name__ + "/")
        n_subspace_anchors = eval_agent.agent.agents[1][0].n_anchors

        # If the number of steps is specified, only plot on thresholds
        if n_subspace_anchors == 3:
            if n_steps is not None:
                if (len(self.current_thresholds) == 0) or (n_steps < self.current_thresholds[0]):
                    return
                else:
                    del self.current_thresholds[0]
        else:
            # To avoid spamming the logs
            if n_steps is None:
                logger.message(f"Can't visualize the subspace, as it does not have exactly 3 different anchors: it currently has {n_subspace_anchors} anchors)")
            return

        message_steps_str = f" at step {n_steps:,d}" if n_steps is not None else ""
        logger.message(f"Preparing to plot the subspace" + message_steps_str)

        # Compute alphas and rewards
        points_left = generate_left_edge_points(self.num_points)
        points_lower = generate_lower_edge_points(self.num_points)

        axis_equations_points = []
        i = 0
        slope = -np.sqrt(3)/3
        for point in points_left:
            new_slope, b = find_axis_through_point(point, slope)
            axis_equations_points.append((new_slope, b))

        intersection_points = []
        i, j = 0, 0
        for i in range(len(axis_equations_points)):
            for j in range(len(points_lower)):
                slope, b = axis_equations_points[i]
                x_coord = points_lower[j][0]
                intersection_point_val = intersection_point(slope, b, x_coord)
                # Check if the intersection point is not already in points_left or points_lower
                if (intersection_point_val not in points_left) and (intersection_point_val not in points_lower):
                    # Check if the intersection point lies in the positive quadrant
                    if intersection_point_val[0] > 0 and intersection_point_val[1] > 0:
                        intersection_points.append(intersection_point_val)

        # Extract x and y coordinates from points
        all_points = points_left + points_lower + intersection_points
        x_points = [point[0] for point in all_points]
        y_points = [point[1] for point in all_points]

        alpha_reward_list = []  # List to store tuples of alphas and rewards
        cpt = 0

        # Compute the reward of a given point (tip of the triangle)
        alpha = torch.Tensor([0, 0, 1])
        reward = evaluate_agent(eval_agent, alpha)
        alpha_reward_list.append((alpha, reward))

        logger.message(f"Evaluating the rewards for different policies ({self.num_points} points sampled per edge)")
        _plotting_start_time = time.time()
        for i in range(len(x_points)):
            if is_inside_triangle([x_points[i], y_points[i]], [0, 0], [1, 0], [0.5, np.sqrt(3) / 2]):
                if x_points[i] != 0.5 and y_points[i] != np.sqrt(3)/2:
                    alpha = get_alphas_from_point(x_points[i], y_points[i])
                    reward = evaluate_agent(eval_agent, alpha)
                    alpha_reward_list.append((alpha, reward))
                    cpt += 1

        logger.message("Time elapsed: " + str(round(time.time() - _plotting_start_time, 0)) + " sec")

        # Plot the triangle with the points
        if self.is_interactive:
            self.plot_interactive_triangle_with_multiple_points(alpha_reward_list, logger, info=info, n_steps=n_steps)
        self.plot_triangle_with_multiple_points(alpha_reward_list, logger, info=info, n_steps=n_steps)



    def plot_triangle_with_multiple_points(self, alpha_reward_list, logger, info={}, n_steps=None, **kwargs):
        """
        Plot a triangle with vertices representing policies

        Parameters:
        - coefficients_list (list of lists): List of coefficients (a1, a2, a3) used to calculate the new point.
        - rewards_list (list of float): List of reward values for each point.
        """
        logger.message(f"Saving the subspace rewards figure")
        _plotting_start_time = time.time()

        # Plot the vertices of the equilateral triangle
        triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

        fig = plt.figure(figsize=(9, 7))

        # Plot policy vertices
        # π1 is the first anchor of the subspace, π2 is the second anchor, and π3 the third one
        plt.text(triangle_vertices[0, 0] - 0.06, triangle_vertices[0, 1] - 0.04, "π1", fontsize=10)
        plt.text(triangle_vertices[1, 0] + 0.01, triangle_vertices[1, 1] - 0.04, "π2", fontsize=10)
        plt.text(triangle_vertices[2, 0] - 0.02, triangle_vertices[2, 1] + 0.02, "π3", fontsize=10)

        # Display the similarities of the anchors
        plt.text(-0.05, 0.75, f"{info['anchors_similarities']}\nArea = {info['subspace_area']:.2f}", fontsize=10)

        # Normalize rewards for colormap
        _, rewards_list = zip(*alpha_reward_list)
        norm = plt.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))
        norm = mcolors.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))
        # For CartPole: set the colorbar between 0 and 500
        if self.env_name == "CartPoleContinuous-v1":
            norm = mcolors.Normalize(vmin=0, vmax=500)

        # Choose a colormap that covers the entire range of rewards
        cmap = plt.get_cmap('RdBu_r')

        # Plot the points with adjusted positions and transparency
        for coeffs, reward in alpha_reward_list:
            new_point = np.dot(np.array(coeffs), triangle_vertices[:3])
            # jitter = np.random.normal(0, 0.01, 2)  # Add a small random jitter to avoid superposition
            # plt.scatter(new_point[0] + jitter[0], new_point[1] + jitter[1], c=reward, cmap=cmap, norm=norm, alpha=0.5, s=250)
            plt.scatter(new_point[0], new_point[1], c=reward, cmap=cmap, norm=norm, s=75)

        # Plot the best estimated policy
        if ("best_alpha" in info) and (info["best_alpha"] is not None):
            best_alpha = info["best_alpha"]
            best_alpha_length = len(best_alpha)
            if best_alpha_length < 3:
                best_alpha = torch.cat((best_alpha, torch.zeros(3 - best_alpha_length)))
            best_point = get_point_from_alphas(np.array(best_alpha.tolist()).reshape(1, -1), triangle_vertices)
            plt.scatter(best_point[0], best_point[1], c="yellow", marker="*", edgecolors="black", linewidths=0.5, s=150, label=f"Best estimated policy (reward = {info['best_alpha_reward']:.2f})")

        # Set axis limits and labels
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.05)

        # Add color bar legend
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca())
        cbar.set_label("Reward")

        # Add legend
        if n_steps is None:
            plt.legend(loc="upper right")

        title_steps_str = f" at step {n_steps:,d}" if n_steps is not None else ""
        plt.title(f"{self.algorithm_name} Subspace Rewards for {self.env_name}" + title_steps_str)

        # Then save the plot using the save_directory
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_path_steps_str = f"_step_{n_steps:,d}" if n_steps is not None else ""
        save_path = os.path.join(self.output_path, f"{self.env_name}_{self.algorithm_name}_Subspace_Rewards_{date_time}" + save_path_steps_str + ".png")
        plt.savefig(save_path)

        logger.message("Time elapsed: " + str(round(time.time() - _plotting_start_time, 0)) + " sec")


    def plot_interactive_triangle_with_multiple_points(self, alpha_reward_list, logger, info={}, **kwargs):
        """
        Plot a triangle with vertices representing policies and a new point calculated as a weighted sum of policies.
        Color the new point based on its reward value.

        Parameters:
        - coefficients_list (list of lists): List of coefficients (a1, a2, a3) used to calculate the new point.
        - rewards_list (list of torch.Tensor): List of reward tensors for each point.
        """
        logger.message(f"Plotting the interactive figure of the subspace rewards")
        _plotting_start_time = time.time()
        
        # Generate the vertices of the equilateral triangle
        triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

        # Create trace for policy vertices
        policy_vertices = go.Scatter(
            x=[triangle_vertices[0, 0], triangle_vertices[1, 0], triangle_vertices[2, 0]],
            y=[triangle_vertices[0, 1], triangle_vertices[1, 1], triangle_vertices[2, 1]],
            mode="text",
            text=["π1", "π2", "π3"],
            textposition=["bottom center", "bottom center", "top center"],
            textfont=dict(size=20),
            showlegend=False,  # Set this to False to hide from legend
        )

        # Create trace for new points
        _, rewards_list = zip(*alpha_reward_list)
        # Convert Torch tensors to standard Python numbers
        rewards_list = [reward.item() for reward in rewards_list]

        min_reward = min(rewards_list)
        max_reward = max(rewards_list)

        # Create trace for new points with the updated colorbar attributes
        new_points = []
        for coeffs, reward in alpha_reward_list:
            reward = reward.item()
            new_point = np.dot(np.array(coeffs), triangle_vertices[:3])
            hover_text = f"Reward: {reward:.2f}<br>Coordinates: ({new_point[0]:.2f}, {new_point[1]:.2f})<br>Alphas: {coeffs}"
            new_points.append(go.Scatter(
                x=[float(new_point[0])],
                y=[float(new_point[1])],
                mode="markers",
                marker=dict(
                    color=[reward],  # Enclose reward in a list to define its color based on the 'RdBu' scale
                    size=10,
                    colorbar=dict(
                        title="Reward",
                        tickvals=[min_reward, max_reward],
                        ticktext=[f'{min_reward:.2f}', f'{max_reward:.2f}']
                    ),
                    colorscale="RdBu_r",  # Use the built-in Red-Blue color scale
                    cmin=min_reward,  # Explicitly set the min for color scaling
                    cmax=max_reward,  # Explicitly set the max for color scaling
                    showscale=True  # Ensure that the colorscale is shown
                ),
                text=[hover_text],
                hoverinfo='text',
                showlegend=False,
            ))

        # Plot the best estimated policy
        if ("best_alpha" in info) and (info["best_alpha"] is not None):
            coeffs = info["best_alpha"]
            best_alpha_length = len(coeffs)
            if best_alpha_length < 3:
                coeffs = torch.cat((coeffs, torch.zeros(3 - best_alpha_length)))
            best_point = get_point_from_alphas(np.array(coeffs.tolist()).reshape(1, -1), triangle_vertices)
            reward = info["best_alpha_reward"]

            hover_text = f"Reward: {reward:.2f}<br>Coordinates: ({best_point[0]:.2f}, {best_point[1]:.2f})<br>Alphas: {coeffs}"
            new_points.append(go.Scatter(
                x=[float(best_point[0])],
                y=[float(best_point[1])],
                mode="markers",
                marker=dict(
                    color="yellow",
                    line=dict(width=0.5,
                        color="DarkSlateGrey"),
                    size=20,
                    symbol="star",
                ),
                name="Best estimated policy",
                text=[hover_text],
                hoverinfo='text',
                showlegend=True,
            ))

        # Create layout
        layout = go.Layout(
            title=f"{self.algorithm_name} Subspace Rewards for {self.env_name}",
            xaxis=dict(title="X-axis", range=[-0.1, 1.1]),
            yaxis=dict(title="Y-axis", range=[-0.1, 1.05]),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            margin=dict(l=60, r=20, t=20, b=20),
            plot_bgcolor="white",
            showlegend=True
        )

        logger.message("Time elapsed: " + str(round(time.time() - _plotting_start_time, 0)) + " sec")

        fig = go.Figure(data=[policy_vertices] + new_points, layout=layout)
        fig.show()
