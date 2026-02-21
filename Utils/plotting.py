import matplotlib.pyplot as plt

def plot_histogram_from_dict(class_counts, figsize=(8, 5), title="Class Distribution"):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=figsize)
    bars = plt.bar(classes, counts)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f"{int(height)}", ha="center", va="bottom")

    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    # plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

# def plot_length_distribution(lengths, figsize=(8, 5), title="Length Distribution", bins=None, xlabel="Length", ylabel="Frequency"):
#     if bins is None:
#         bins = min(10, len(set(lengths)))

#     plt.figure(figsize=figsize)
#     plt.hist(lengths, bins=bins, edgecolor="black", alpha=0.7)

#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     # plt.grid(axis="y", linestyle="--", alpha=0.4)
#     plt.grid(False)
#     plt.tight_layout()
#     plt.show()


def plot_length_distribution(lengths, figsize=(8, 5), title="Length Distribution",
                             bins=None, xlabel="Length", ylabel="Frequency", crop_x_axis=False):
    if bins is None:
        bins = min(10, len(set(lengths)))

    plt.figure(figsize=figsize)

    counts, bin_edges, _ = plt.hist(
        lengths,
        bins=bins,
        edgecolor="black",
        alpha=0.7
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if not crop_x_axis:
        plt.xlim(0, max(lengths) * 1.1)
    plt.grid(False)
    plt.tight_layout()
    plt.show()