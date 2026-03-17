import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../data/Depression_cones.csv")
    #df = df[['Iso_width','Iso_distance','Area_root']]
    matrix = df.corr()
    print(matrix)
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", linewidths=0.5,cmap='coolwarm',center=0,)
    ax.figure.tight_layout()
    plt.title("Correlation Heatmap Depression Cones Dataset")
    plt.savefig("correlation_depression_cones.png",bbox_inches='tight')
