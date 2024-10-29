from constants import NUM_STEPS
import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_heatmaps(opinions):
    opinions_matrix = []
    for step in range(1, NUM_STEPS // 5):
        opinions_matrix.append([opinions[step][node] for node in opinions[step]])
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    for i, ax in enumerate(axes.flatten()):
        sns.heatmap(opinions_matrix[i], ax=ax, cmap='coolwarm')
        ax.set_title(f'Step {i+1}')
    
    plt.tight_layout()
    plt.show()
    fig.savefig('similarity_heatmaps.png')

def plot_opinions_evolution(equal_opinions):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(equal_opinions.keys(), equal_opinions.values(), linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Number of nodes with equal opinions')
    plt.show()
    fig.savefig('opinions_evolution.png')