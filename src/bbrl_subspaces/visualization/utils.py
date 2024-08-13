# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np

from bbrl.workspace import Workspace



def evaluate_agent(eval_agent, alpha):
    workspace = Workspace()
    eval_agent(workspace, t=0, stop_variable="env/done", alphas=alpha)
    rewards = workspace["env/cumulated_reward"][-1]
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
    # We know that the slope is the same as the given slope, so b = y - mx
    b = y_h - slope * x_h
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



def get_alphas_from_point(x, y):
    triangle_vertices = [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]
    A = np.vstack([triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]]).T
    A = np.vstack([A, np.ones(3)])
    b = np.array([x, y, 1])  

    # Solving the equations system
    alphas = np.linalg.lstsq(A, b, rcond=None)[0]    
    alphas = np.maximum(alphas, 0)  # Ensure that all values are positive
    alphas /= np.sum(alphas)    # Normalize
    return torch.Tensor(alphas)



def is_inside_triangle(point, A, B, C):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(point, A, B) < 0.0
    b2 = sign(point, B, C) < 0.0
    b3 = sign(point, C, A) < 0.0
    return ((b1 == b2) and (b2 == b3))



def get_point_from_alphas(alphas, vertices):
    # alphas is a 1xN vector, vertices is a stack of row vectors, whose each row is the coordinates of a tip of the subspace
    # returns a point of dimension (2,)
    return (alphas @ vertices).reshape(-1)