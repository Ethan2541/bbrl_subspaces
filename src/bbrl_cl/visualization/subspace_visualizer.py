import os
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import numpy as np

from datetime import datetime

from .utils import evaluate_agent, find_axis_through_point, generate_left_edge_points, generate_lower_edge_points, get_alphas_from_point, intersection_point, is_inside_triangle



class SubspaceVisualizer:
    def __init__(self, params, output_path="./figures/"):
        self.cfg = params
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path


    def plot_subspace(self, eval_agent, logger):
        logger = logger.get_logger(type(self).__name__)
        logger.message("Preparing to plot the subspace")

        # Compute alphas and rewards
        points_left = generate_left_edge_points(self.cfg.num_points)
        points_lower = generate_lower_edge_points(self.cfg.num_points)

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
                # Check if the intersection point is not already in points or points2
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

        logger.message(f"Evaluating the rewards for {self.cfg.num_points} different policies")
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
        self.plot_triangle_with_multiple_points(alpha_reward_list, logger)


    def plot_triangle_with_multiple_points(self, alpha_reward_list, logger):
        """
        Plot a triangle with vertices representing policies

        Parameters:
        - coefficients_list (list of lists): List of coefficients (a1, a2, a3) used to calculate the new point.
        - rewards_list (list of float): List of reward values for each point.
        """
        logger.message(f"Plotting the triangle frame")
        _plotting_start_time = time.time()

        # Plot the vertices of the equilateral triangle
        triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2],[0,0]])
        plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-')

        # Plot policy vertices
        plt.text(triangle_vertices[0, 0] - 0.05, triangle_vertices[0, 1] - 0.03, 'p1', fontsize=10)
        plt.text(triangle_vertices[1, 0] + 0.05, triangle_vertices[1, 1] - 0.03, 'p2', fontsize=10)
        plt.text(triangle_vertices[2, 0], triangle_vertices[2, 1] + 0.03, 'p3', fontsize=10)

        # norm = mcolors.Normalize(vmin=0, vmax=500) #TO TO HAVE THE VALUES FROM 0 TO 500
        _, rewards_list = zip(*alpha_reward_list)

        # Normalize rewards for colormap
        norm = plt.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))
        norm = mcolors.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))

        # Choose a colormap that covers the entire range of rewards
        cmap = plt.get_cmap('RdBu_r')

        # Plot the points with adjusted positions and transparency
        for coefs, reward in (alpha_reward_list):
            new_point = np.dot(np.array(coefs), triangle_vertices[:3])
            # jitter = np.random.normal(0, 0.01, 2)  # Add a small random jitter to avoid superposition
            # plt.scatter(new_point[0] + jitter[0], new_point[1] + jitter[1], c=reward, cmap=cmap, norm=norm, alpha=0.5, s=250)
            plt.scatter(new_point[0], new_point[1], c=reward, cmap=cmap, norm=norm, s=60)

        # Set axis limits and labels
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)

        # Add color bar legend
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca())
        cbar.set_label('Reward')

        # Add legend
        # Define custom legend elements
        legend_elements = [
            mlines.Line2D([], [], color='black', linewidth=1, label='Triangle Edges')
        ]

        # Add legend
        plt.legend(handles=legend_elements)

        # Then save the plot using the save_directory
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(self.output_path, f"subspace_{date_time}.png")
        plt.savefig(save_path)

        logger.message("Time elapsed: " + str(round(time.time() - _plotting_start_time, 0)) + " sec")
        plt.show()