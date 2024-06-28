import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from datetime import datetime

import os
import sys
import numpy as np

import hydra
from omegaconf import DictConfig

from bbrl_algos.models.loggers import Logger
# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../algos/dqn')
from dqnv2 import run_dqn

matplotlib.use("TkAgg")


def dynamic_histograms(filename):
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the file
    if filename == "distances.txt":
        distance_file_path = os.path.join(script_directory, filename)

        # Read the file
        with open(distance_file_path, "r") as distance_file:
            distance_data = [float(line.strip()) for line in distance_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=distance_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs de la distance',
            xaxis=dict(title='Intervalles de valeurs pris par la distance'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()



    elif filename == "gradient_norm.txt":
        grad_file_path = os.path.join(script_directory, filename)

        with open(grad_file_path, "r") as grad_file:
            grad_data = [float(line.strip()) for line in grad_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=grad_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs du gradient',
            xaxis=dict(title='Intervalles de valeurs pris par le gradient'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()

    elif filename == "loss_values.txt":
        loss_file_path = os.path.join(script_directory, filename)

        with open(loss_file_path, "r") as loss_file:
            loss_data = [float(line.strip()) for line in loss_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=loss_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs de la loss',
            xaxis=dict(title='Intervalles de valeurs pris par la loss'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()
    
    elif filename == "facteurs_k.txt":
        k_file_path = os.path.join(script_directory, filename)

        with open(k_file_path, "r") as k_file:
            k_data = [float(line.strip()) for line in k_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=k_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs du facteur k',
            xaxis=dict(title='Intervalles de valeurs pris par le facteur k'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()

    elif filename == "differences_grad_distances.txt":
        diff_file_path = os.path.join(script_directory, filename)

        with open(diff_file_path, "r") as diff_file:
            diff_data = [float(line.strip()) for line in diff_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=diff_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs de la difference entre le facteur k et learning rate',
            xaxis=dict(title='Intervalles de valeurs pris par la difference entre le facteur k et learning rate'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()



def histograms(filename):

    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the file
    if(filename == "distances.txt"):
        distance_file_path = os.path.join(script_directory, filename)

        # Read the file
        with open(distance_file_path, "r") as distance_file:
            distance_data = [float(line.strip()) for line in distance_file.readlines()]

        # Convert the distance values
        distance_values = [float(value) for value in distance_data]
        #distance_values = [value/10 for value in distance_values]

        num_bins = 20

        # Plot the histogramm 
        plt.figure(figsize=(10, 6))
        plt.hist(distance_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la distance')
        plt.xlabel('Intervalles de valeurs pris par la distance')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..","..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    if(filename == "gradient_norm.txt"):
        grad_file_path = os.path.join(script_directory, filename)

        with open(grad_file_path, "r") as grad_file:
            grad_data = [float(line.strip()) for line in grad_file.readlines()]

        grad_values = [float(value) for value in grad_data]
        #grad_values = [value/10 for value in grad_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(grad_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs du gradient')
        plt.xlabel('Intervalles de valeurs pris par le gradient')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..","..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    if(filename == "loss_values.txt"):
        loss_file_path = os.path.join(script_directory, filename)

        with open(loss_file_path, "r") as loss_file:
            loss_data = [float(line.strip()) for line in loss_file.readlines()]

        loss_values = [float(value) for value in loss_data]
        #grad_values = [value/10 for value in grad_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(loss_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la loss')
        plt.xlabel('Intervalles de valeurs pris par la loss')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..","..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()


    if(filename == "facteurs_k.txt"):
        k_file_path = os.path.join(script_directory, filename)

        with open(k_file_path, "r") as k_file:
            #loss_data = [line.split(", ")[1].split(": ")[1] for line in loss_file.readlines()]
            k_data = [float(line.strip()) for line in k_file.readlines()]

        k_values = [float(value) for value in k_data]
        #loss_values = [value/10 for value in loss_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(k_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs du facteur k')
        plt.xlabel('Intervalles de valeurs pris par le facteur k')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..","..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    if(filename == "differences_grad_distances.txt"):
        diff_file_path = os.path.join(script_directory, filename)

        with open(diff_file_path, "r") as loss_file:
            #loss_data = [line.split(", ")[1].split(": ")[1] for line in loss_file.readlines()]
            diff_data = [float(line.strip()) for line in loss_file.readlines()]

        diff_values = [float(value) for value in diff_data]
        #loss_values = [value/10 for value in loss_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(diff_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la difference entre le facteur k et learning rate')
        plt.xlabel('Intervalles de valeurs pris par la difference entre le facteur k et learning rate')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..","..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    plt.show()


@hydra.main(
    config_path="../algos/dqn/configs/",
    config_name="dqn_cartpole.yaml",
    #config_name="dqn_lunar_lander.yaml", #cartpole
)  # , version_base="1.3")

def main(cfg_raw: DictConfig):

    logger = Logger(cfg_raw)
    run_dqn(cfg_raw, logger)
    # then stocking the k values in a txt file
    #print(is_grad_norm_proportional_to_distance(cfg)) # VRAI POUR MAX_GRAD_NORM = 1000 et a 0.02 pret

    # Call the histograms function with the desired file you want to plot
    #histograms("distances.txt")
    #histograms("gradient_norm.txt")
    #histograms("loss_values.txt")

    #histograms("differences_grad_distances.txt")
    #histograms("facteurs_k.txt")

    dynamic_histograms("distances.txt")
    dynamic_histograms("gradient_norm.txt")
    dynamic_histograms("loss_values.txt")
    #dynamic_histograms("differences_grad_distances.txt")
    #dynamic_histograms("facteurs_k.txt")

if __name__ == "__main__":
    main()