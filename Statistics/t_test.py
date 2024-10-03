import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

def load_data(path):
    """
    Load the data from the specified path.

    Args:
        path (str): the path to the data

    Returns:
        data (pd.DataFrame): the loaded data
    """
    data = pd.read_csv(path, sep=';')
    return data

def filter_volatile_acidity(data):
    """
    Filter the data for volatile acidity values for quality levels 3+4 and 7+8.

    Args:
        data (pd.DataFrame): the data to be filtered

    Returns:
        x, y: the filtered data
    """
    x = data['volatile acidity'][(data['quality'] == 3) | (data['quality'] == 4)]
    y = data['volatile acidity'][(data['quality'] == 7) | (data['quality'] == 8)]
    return x, y

def perform_ttest(x, y):
    """
    Perform a two-sample t-test to compare the distribution of volatile acidity.

    Args:
        x: sample 1
        y: sample 2

    Returns:
        ttest: t-test result
    """
    ttest = stats.ttest_ind(x, y)
    return ttest

def plot_volatile_acidity(x, y, ttest):
    """
    Plot the distribution of volatile acidity values for quality levels 3+4 and 7+8.

    Args:
        x: sample 1
        y: sample 2
        ttest: t-test result
    """
    plt.subplot(1, 2, 1)  
    plt.plot(np.random.randn(len(x))/30, x, 'o', np.random.randn(len(y))/30 + 1, y, 'o', markeredgecolor='k')
    plt.xlim([-1, 2])
    plt.xticks([0, 1], labels=['Quality 3+4', 'Quality 7+8'])
    plt.ylabel('Volatile Acidity')
    plt.title(f't={ttest[0]:.2f}, p={ttest[1]:.5f}')

def plot_quality_counts(data):
    """
    Plot the distribution of quality ratings.

    Args:
        data (pd.DataFrame): the data to be plotted
    """
    counts = data['quality'].value_counts()
    plt.subplot(1, 2, 2)  
    plt.bar(list(counts.keys()), counts)
    plt.xlabel('Quality Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Quality Ratings')

def main():
    path = "./Statistics/data.csv"
    data = load_data(path)
    x, y = filter_volatile_acidity(data)
    ttest = perform_ttest(x, y)
    plt.figure(figsize=(10, 5))
    plot_volatile_acidity(x, y, ttest)
    plot_quality_counts(data)
    plt.tight_layout()  
    plt.show()

if __name__ == "__main__":
    main()

