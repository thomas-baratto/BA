import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mc
import os
import network
import numpy as np
import pandas as pd

def visualize_ellipse(data):
    os.makedirs("./ellipse", exist_ok=True)
    colors = mc.TABLEAU_COLORS
    colors = list(colors.values())
    counter = 0
    max_width = 0
    max_distance = 0
    for group in data:
        fig, ax = plt.subplots(figsize=(15, 15))
        for row in group:
            Iso_width, Iso_distance, Area, Isotherm = row[9:]
            if max_width < Iso_width: max_width = Iso_width
            if max_distance < Iso_distance: max_distance = Iso_distance
            ellipse = patches.Ellipse((0,0),width=Iso_distance,height=Iso_width,label = Isotherm,fill=False,color=colors[int(Isotherm)])
            ax.add_patch(ellipse)

        ax.set_xlim(-max_distance, max_distance)
        ax.set_ylim(-max_width, max_width)
        ax.set_aspect("equal")
        plt.grid()
        plt.legend()
        plt.savefig(fname=f"ellipse/%s.svg"%counter,format = 'svg')
        plt.close(fig)
        counter+=1

def visualize_gplume(data):
    os.makedirs("./gplumes", exist_ok=True)
    colors = mc.TABLEAU_COLORS
    colors = list(colors.values())
    counter = 0
    max_width = 0
    max_distance = 0
    for group in data:
        fig, ax = plt.subplots(figsize=(15, 15))
        for row in group:
            Iso_width, Iso_distance, Area, Isotherm = row[9:]
            if max_width < Iso_width: max_width = Iso_width
            if max_distance < Iso_distance: max_distance = Iso_distance
            ellipse = patches.Ellipse((0,0),width=Iso_distance,height=Iso_width,label = Isotherm,fill=False,color=colors[int(Isotherm)])
            ax.add_patch(ellipse)

        ax.set_xlim(-max_distance, max_distance)
        ax.set_ylim(-max_width, max_width)
        ax.set_aspect("equal")
        plt.grid()
        plt.legend()
        plt.savefig(fname=f"gplumes/%s.svg"%counter,format = 'svg')
        plt.close(fig)
        counter+=1


if __name__ == "__main__":

    df = pd.read_csv('../Daten/Clean_Results_Isotherm.csv')
    data = [group.values.tolist() for _, group in df.groupby('Temp_diff_real')]

    visualize_ellipse(data[-3:-1])