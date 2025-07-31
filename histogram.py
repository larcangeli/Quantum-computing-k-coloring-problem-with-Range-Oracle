import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Data provided by the user, with model names updated
# data = {
#     'ibm_osaka': {
#         'Minimum Depth': [0.5947265625, 0.5693359375, 0.5693359375, 0.5791015625, 0.5966796875],
#         'Minimum Width': [0.5126953125, 0.5322265625, 0.51953125, 0.4912109375, 0.4970703125],
#         'Balanced': [0.5595703125, 0.572265625, 0.5361328125, 0.6044921875, 0.572265625],
#         'Fixed Original': [0.5654296875, 0.5634765625, 0.5703125, 0.54296875, 0.5439453125]
#     },
#     'ibm_kyoto': {
#         'Minimum Depth': [0.5771484375, 0.5771484375, 0.5712890625, 0.59375, 0.5693359375],
#         'Minimum Width': [0.5673828125, 0.5634765625, 0.5302734375, 0.544921875, 0.5234375],
#         'Balanced': [0.5380859375, 0.55078125, 0.5498046875, 0.560546875, 0.5322265625],
#         'Fixed Original': [0.5693359375, 0.5693359375, 0.53515625, 0.546875, 0.546875]
#     },
#     'ibm_brisbane': {
#         'Minimum Depth': [0.8486328125, 0.8466796875, 0.873046875, 0.85546875, 0.841796875],
#         'Minimum Width': [0.7734375, 0.80078125, 0.7958984375, 0.8125, 0.8076171875],
#         'Balanced': [0.8330078125, 0.845703125, 0.830078125, 0.8349609375, 0.8046875],
#         'Fixed Original': [0.8330078125, 0.826171875, 0.8310546875, 0.82421875, 0.7998046875]
#     },
#     'ibm_sherbrooke': {
#         'Minimum Depth': [0.8173828125, 0.8232421875, 0.8193359375, 0.8291015625, 0.8115234375],
#         'Minimum Width': [0.783203125, 0.7822265625, 0.79296875, 0.7978515625, 0.8056640625],
#         'Balanced': [0.8056640625, 0.80859375, 0.8173828125, 0.7978515625, 0.802734375],
#         'Fixed Original': [0.796875, 0.8232421875, 0.8095703125, 0.798828125, 0.767578125]
#     }
# }


def plot_simulation_data(data, random_guess_chance):
    # Convert values from 0 to 1 scale to 0 to 100 scale
    for model in data:
        for configuration in data[model]:
            data[model][configuration] = [x * 100 for x in
                                          data[model][configuration]]

    plt.rcParams.update({'font.size': 20})


    # Configuration for histogram plot
    labels = list(data.keys())  # Noise model names
    model_names = list(data[labels[0]].keys())  # Model names
    x = np.arange(len(labels))  # Label locations
    width = 0.2  # Width of the bars
    # Diverse colors for each model
    colors = ['skyblue', 'lightgreen', 'salmon', 'violet']

    formal_name = {"simple": "Minimum depth oracle",
                   "balanced": "Balanced oracle",
                   "original": "Original paper oracle",
                   "minimal": "Minimum width oracle"}

    # Function to calculate mean and 95% confidence interval
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    fig, ax = plt.subplots(figsize=(12, 8))
    max_value = 0
    for i, model_name in enumerate(model_names):
        means = []
        cis = []
        for label in labels:
            mean, ci_low, ci_high = mean_confidence_interval(data[label][model_name])
            means.append(mean)
            cis.append((mean-ci_low, ci_high-mean))
            max_value = max(max_value, mean + ci_high)  # Find max value to set yticks

        # Plotting the bars for each model
        cis = np.array(cis).T
        ax.bar(x + i*width, means, width, label=formal_name[model_name], yerr=cis,
               capsize=5, color=colors[i])
    # Customize x-axis
    ax.set_xticks(x + width*(len(model_names)-1)/2)
    ax.set_xticklabels(labels)

    # Adding labels and titles
    ax.set_xlabel('Noise Models')
    ax.set_ylabel('Correct Result Probability (%)')
    ax.legend(title="Model")

    # Add a horizontal line for random correct guess probability
    ax.axhline(y=random_guess_chance * 100, color='r', linestyle='--',
               label='Random Correct Guess Probability (' +
               str(random_guess_chance * 100) + '%)')
    ax.legend()

    # Set y-ticks dynamically based on the data
    ytick_spacing = 5  # Adjust this as needed for finer granularity
    ax.set_yticks(np.arange(0, 100 + ytick_spacing, step=ytick_spacing))

    plt.show()
    plt.close()
