import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

# matplotlib.use('Agg')  # No GUI
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# Enable LaTeX
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif" # Matches standard paper fonts like Times/Palatino
# })

from src.conf import EVAL_ROUND
from src.utils import load, verify_metrics


def w_heatmaps(W1, W2, threshold=0.001):
    """
    Draws light-colored heatmaps of two doubly stochastic matrices.
    - Elements below threshold: Black cell, White text.
    - Diagonal elements: White cell, No text.
    - Other elements: Light colormap, Black text.
    - Aspect ratio: Rectangular (length = 2 * width).
    """
    W1 = np.array(
        [
            [0.0, 0.0, 0.13212004, 0.2127421, 0.0, 0.0, 0.11443867, 0.23008352, 0.17763015, 0.13298692],
            [0.0, 0.0, 0.26018556, 0.1631708, 0.20866143, 0.0, 0.1606299, 0.0, 0.0, 0.20734599],
            [0.13212004, 0.26018556, 0.0, 0.09974875, 0.08232586, 0.0, 0.10634067, 0.0, 0.21307175, 0.10620618],
            [0.2127421, 0.1631708, 0.09974875, 0.0, 0.09480792, 0.25296527, 0.05609525, 0.0, 0.0, 0.12046996],
            [0.0, 0.20866143, 0.08232586, 0.09480792, 0.0, 0.16723207, 0.07927795, 0.2653364, 0.10236621, 0.0],
            [0.0, 0.0, 0.0, 0.25296527, 0.16723207, 0.0, 0.17574189, 0.0, 0.40405657, 0.0],
            [
                0.11443867,
                0.1606299,
                0.10634067,
                0.05609525,
                0.07927795,
                0.17574189,
                0.0,
                0.13808644,
                0.10287641,
                0.06651608,
            ],
            [0.23008352, 0.0, 0.0, 0.0, 0.2653364, 0.0, 0.13808644, 0.0, 0.0, 0.36648329],
            [0.17763015, 0.0, 0.21307175, 0.0, 0.10236621, 0.40405657, 0.10287641, 0.0, 0.0, 0.0],
            [0.13298692, 0.20734599, 0.10620618, 0.12046996, 0.0, 0.0, 0.06651608, 0.36648329, 0.0, 0.0],
        ]
    )
    W2 = np.array(
        [
            [0.0, 0.0, 0.0, 0.59714308, 0.0, 0.0, 0.40285507, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.31094389, 0.0, 0.25823903, 0.0, 0.20986632, 0.0, 0.0, 0.22094606],
            [0.0, 0.31094389, 0.0, 0.08100137, 0.0, 0.0, 0.10830173, 0.0, 0.41153176, 0.08821858],
            [0.59714308, 0.0, 0.08100137, 0.0, 0.07972792, 0.15490081, 0.0, 0.0, 0.0, 0.08722791],
            [0.0, 0.25823903, 0.0, 0.07972792, 0.0, 0.12165412, 0.08361213, 0.45677527, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.15490081, 0.12165412, 0.0, 0.13496438, 0.0, 0.58847433, 0.0],
            [0.40285507, 0.20986632, 0.10830173, 0.0, 0.08361213, 0.13496438, 0.0, 0.0, 0.0, 0.06040243],
            [0.0, 0.0, 0.0, 0.0, 0.45677527, 0.0, 0.0, 0.0, 0.0, 0.54321383],
            [0.0, 0.0, 0.41153176, 0.0, 0.0, 0.58847433, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.22094606, 0.08821858, 0.08722791, 0.0, 0.0, 0.06040243, 0.54321383, 0.0, 0.0],
        ]
    )
    # Create a custom light colormap
    # 'set_bad' configures how NaNs (the diagonal) are rendered
    light_cmap = ListedColormap(["#f8f8ff", "#d3d3d3", "#b0c4de", "#add8e6", "#87ceeb", "#4682b4"])
    light_cmap.set_bad(color="white")

    # Create copies to safely modify for plotting without changing original data
    W1_plot = W1.copy()
    W2_plot = W2.copy()

    # Set diagonals to NaN (Not a Number)
    # This ensures they are ignored by the " < threshold" check and colored white by set_bad
    np.fill_diagonal(W1_plot, np.nan)
    np.fill_diagonal(W2_plot, np.nan)

    # Prepare masks for elements below the threshold (ignoring NaNs/Diagonals)
    # Note: 'zeros_like' creates a boolean False array.
    # Comparisons with NaN always return False, so the diagonal is naturally excluded from the black mask.
    mask1 = W1_plot < threshold
    mask2 = W2_plot < threshold

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    ax1, ax2 = axes

    # --- Plotting Helper Function ---
    def plot_matrix(ax, matrix_data, mask_data, original_data, title_text):
        # 1. Base Layer: The Colormap (Handles Normal values + White Diagonals via NaN)
        ax.imshow(matrix_data, cmap=light_cmap, interpolation="nearest")

        # 2. Overlay Layer: Black Cells (Values < Threshold)
        # alpha=mask_data ensures only the 'True' parts of the mask are painted black
        ax.imshow(mask_data, cmap=ListedColormap(["black"]), interpolation="nearest", alpha=mask_data.astype(float))

        # 3. Settings
        ax.set_aspect(0.5)  # Rectangular cells
        ax.axis("off")
        ax.text(4, 11, title_text, fontsize=20, weight="bold")

        # 4. Text Annotations
        rows, cols = original_data.shape
        for i in range(rows):
            for j in range(cols):
                # RULE: Skip text for diagonals
                if i == j:
                    continue

                value = original_data[i, j]

                # RULE: White text on Black background, Black text otherwise
                if value < threshold:
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
                else:
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=8)

    # --- Render Plots ---
    plot_matrix(ax1, W1_plot, mask1, W1, r"$\mathbf{W^r}$")
    plot_matrix(ax2, W2_plot, mask2, W2, r"$\mathbf{W^r_p}$")

    plt.tight_layout()
    plt.show()


def w_heatmaps__(W1, W2, threshold=0.001):
    """
    Draws light-colored heatmaps of two doubly stochastic matrices W1 and W2.
    Elements below the threshold are displayed as black cells with white text.
    The cells have a rectangle shape (length = 2 * width).

    Parameters:
        W1 (numpy.ndarray): The first doubly stochastic matrix.
        W2 (numpy.ndarray): The second doubly stochastic matrix.
        threshold (float): Threshold below which elements are shown in black.
    """
    W1 = np.array(
        [
            [
                0.000000000000,
                0.152116863508,
                0.252385800688,
                0.114642547510,
                0.082588969393,
                0.031505508722,
                0.089013398335,
                0.076108874263,
                0.134494017906,
                0.067144019676,
            ],
            [
                0.152116863508,
                0.000000000000,
                0.000000000000,
                0.156983734893,
                0.173624063510,
                0.000000000000,
                0.000000000000,
                0.207261241191,
                0.129182758515,
                0.180831338383,
            ],
            [
                0.252385800688,
                0.000000000000,
                0.000000000000,
                0.213288145987,
                0.174290630302,
                0.000000000000,
                0.140170783753,
                0.000000000000,
                0.079065662692,
                0.140798976578,
            ],
            [
                0.114642547510,
                0.156983734893,
                0.213288145987,
                0.000000000000,
                0.136839533228,
                0.143321256099,
                0.000000000000,
                0.176184620638,
                0.000000000000,
                0.058740161644,
            ],
            [
                0.082588969393,
                0.173624063510,
                0.174290630302,
                0.136839533228,
                0.000000000000,
                0.000000000000,
                0.134969670969,
                0.113539672042,
                0.081080119415,
                0.103067341141,
            ],
            [
                0.031505508722,
                0.000000000000,
                0.000000000000,
                0.143321256099,
                0.000000000000,
                0.000000000000,
                0.281167755607,
                0.155141407743,
                0.251306501854,
                0.137557569976,
            ],
            [
                0.089013398335,
                0.000000000000,
                0.140170783753,
                0.000000000000,
                0.134969670969,
                0.281167755607,
                0.000000000000,
                0.102603292488,
                0.217123168749,
                0.034951930100,
            ],
            [
                0.076108874263,
                0.207261241191,
                0.000000000000,
                0.176184620638,
                0.113539672042,
                0.155141407743,
                0.102603292488,
                0.000000000000,
                0.000000000000,
                0.169160891634,
            ],
            [
                0.134494017906,
                0.129182758515,
                0.079065662692,
                0.000000000000,
                0.081080119415,
                0.251306501854,
                0.217123168749,
                0.000000000000,
                0.000000000000,
                0.107747770869,
            ],
            [
                0.067144019676,
                0.180831338383,
                0.140798976578,
                0.058740161644,
                0.103067341141,
                0.137557569976,
                0.034951930100,
                0.169160891634,
                0.107747770869,
                0.000000000000,
            ],
        ]
    )
    W2 = np.array(
        [
            [
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.176849165520,
                0.224602992315,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.131004000860,
                0.467543841305,
            ],
            [
                0.000000000000,
                0.000000000000,
                0.112009879537,
                0.000000000000,
                0.154869953150,
                0.316598443818,
                0.158870108634,
                0.000000000000,
                0.156299160840,
                0.101352454020,
            ],
            [
                0.000000000000,
                0.112009879537,
                0.000000000000,
                0.374672483150,
                0.123170923072,
                0.161185875296,
                0.000000000000,
                0.131842595036,
                0.097118243909,
                0.000000000000,
            ],
            [
                0.176849165520,
                0.000000000000,
                0.374672483150,
                0.000000000000,
                0.186140442421,
                0.000000000000,
                0.000000000000,
                0.112266592085,
                0.150071316824,
                0.000000000000,
            ],
            [
                0.224602992315,
                0.154869953150,
                0.123170923072,
                0.186140442421,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.207530454189,
                0.103685234853,
                0.000000000000,
            ],
            [
                0.000000000000,
                0.316598443818,
                0.161185875296,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.285778317494,
                0.236437363391,
                0.000000000000,
                0.000000000000,
            ],
            [
                0.000000000000,
                0.158870108634,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.285778317494,
                0.000000000000,
                0.232955301265,
                0.087073458306,
                0.235322814301,
            ],
            [
                0.000000000000,
                0.000000000000,
                0.131842595036,
                0.112266592085,
                0.207530454189,
                0.236437363391,
                0.232955301265,
                0.000000000000,
                0.078967694034,
                0.000000000000,
            ],
            [
                0.131004000860,
                0.156299160840,
                0.097118243909,
                0.150071316824,
                0.103685234853,
                0.000000000000,
                0.087073458306,
                0.078967694034,
                0.000000000000,
                0.195780890374,
            ],
            [
                0.467543841305,
                0.101352454020,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.235322814301,
                0.000000000000,
                0.195780890374,
                0.000000000000,
            ],
        ]
    )

    W1 = np.array(
        [
            [0.0, 0.23160009, 0.0, 0.1020407, 0.09252448, 0.14792331, 0.08031182, 0.0503244, 0.1412419, 0.1540333],
            [0.23160009, 0.0, 0.11019805, 0.0, 0.34698617, 0.19834618, 0.0, 0.11286953, 0.0, 0.0],
            [0.0, 0.11019805, 0.0, 0.22221936, 0.0, 0.0, 0.13228087, 0.10879485, 0.19400622, 0.23250066],
            [0.1020407, 0.0, 0.22221936, 0.0, 0.16163229, 0.1556204, 0.11831694, 0.08660571, 0.1535646, 0.0],
            [0.09252448, 0.34698617, 0.0, 0.16163229, 0.0, 0.15788008, 0.0, 0.15125099, 0.08972599, 0.0],
            [0.14792331, 0.19834618, 0.0, 0.1556204, 0.15788008, 0.0, 0.21928447, 0.05736568, 0.0, 0.06357989],
            [0.08031182, 0.0, 0.13228087, 0.11831694, 0.0, 0.21928447, 0.0, 0.18642312, 0.16938278, 0.093],
            [0.0503244, 0.11286953, 0.10879485, 0.08660571, 0.15125099, 0.05736568, 0.18642312, 0.0, 0.24623064, 0.0],
            [0.1412419, 0.0, 0.19400622, 0.1535646, 0.08972599, 0.0, 0.16938278, 0.24623064, 0.0, 0.00584787],
            [0.1540333, 0.0, 0.23250066, 0.0, 0.0, 0.06357989, 0.093, 0.0, 0.00584787, 0.45103829],
        ]
    )

    W2 = np.array(
        [
            [0.0, 0.14185374, 0.0, 0.0, 0.07181718, 0.08520081, 0.14165413, 0.3185851, 0.10770579, 0.13318335],
            [0.14185374, 0.0, 0.53163538, 0.33518137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.53163538, 0.0, 0.0, 0.0, 0.46836462, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.33518137, 0.0, 0.0, 0.60045638, 0.17298583, 0.0, 0.08470405, 0.0, 0.0],
            [0.07181718, 0.0, 0.0, 0.60045638, 0.0, 0.0, 0.39954362, 0.0, 0.0, 0.0],
            [0.08520081, 0.0, 0.46836462, 0.17298583, 0.0, 0.0, 0.1196072, 0.13008356, 0.17032487, 0.05143309],
            [0.14165413, 0.0, 0.0, 0.0, 0.39954362, 0.1196072, 0.0, 0.09369542, 0.0, 0.24549963],
            [0.3185851, 0.0, 0.0, 0.08470405, 0.0, 0.13008356, 0.09369542, 0.0, 0.0, 0.37293186],
            [0.10770579, 0.0, 0.0, 0.0, 0.0, 0.17032487, 0.0, 0.0, 0.0, 0.72196934],
            [0.13318335, 0.0, 0.0, 0.0, 0.0, 0.05143309, 0.24549963, 0.37293186, 0.72196934, 0.0],
        ]
    )

    W1 = np.array(
        [
            [0.0, 0.0, 0.13212004, 0.2127421, 0.0, 0.0, 0.11443867, 0.23008352, 0.17763015, 0.13298692],
            [0.0, 0.0, 0.26018556, 0.1631708, 0.20866143, 0.0, 0.1606299, 0.0, 0.0, 0.20734599],
            [0.13212004, 0.26018556, 0.0, 0.09974875, 0.08232586, 0.0, 0.10634067, 0.0, 0.21307175, 0.10620618],
            [0.2127421, 0.1631708, 0.09974875, 0.0, 0.09480792, 0.25296527, 0.05609525, 0.0, 0.0, 0.12046996],
            [0.0, 0.20866143, 0.08232586, 0.09480792, 0.0, 0.16723207, 0.07927795, 0.2653364, 0.10236621, 0.0],
            [0.0, 0.0, 0.0, 0.25296527, 0.16723207, 0.0, 0.17574189, 0.0, 0.40405657, 0.0],
            [
                0.11443867,
                0.1606299,
                0.10634067,
                0.05609525,
                0.07927795,
                0.17574189,
                0.0,
                0.13808644,
                0.10287641,
                0.06651608,
            ],
            [0.23008352, 0.0, 0.0, 0.0, 0.2653364, 0.0, 0.13808644, 0.0, 0.0, 0.36648329],
            [0.17763015, 0.0, 0.21307175, 0.0, 0.10236621, 0.40405657, 0.10287641, 0.0, 0.0, 0.0],
            [0.13298692, 0.20734599, 0.10620618, 0.12046996, 0.0, 0.0, 0.06651608, 0.36648329, 0.0, 0.0],
        ]
    )
    W2 = np.array(
        [
            [0.0, 0.0, 0.0, 0.59714308, 0.0, 0.0, 0.40285507, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.31094389, 0.0, 0.25823903, 0.0, 0.20986632, 0.0, 0.0, 0.22094606],
            [0.0, 0.31094389, 0.0, 0.08100137, 0.0, 0.0, 0.10830173, 0.0, 0.41153176, 0.08821858],
            [0.59714308, 0.0, 0.08100137, 0.0, 0.07972792, 0.15490081, 0.0, 0.0, 0.0, 0.08722791],
            [0.0, 0.25823903, 0.0, 0.07972792, 0.0, 0.12165412, 0.08361213, 0.45677527, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.15490081, 0.12165412, 0.0, 0.13496438, 0.0, 0.58847433, 0.0],
            [0.40285507, 0.20986632, 0.10830173, 0.0, 0.08361213, 0.13496438, 0.0, 0.0, 0.0, 0.06040243],
            [0.0, 0.0, 0.0, 0.0, 0.45677527, 0.0, 0.0, 0.0, 0.0, 0.54321383],
            [0.0, 0.0, 0.41153176, 0.0, 0.0, 0.58847433, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.22094606, 0.08821858, 0.08722791, 0.0, 0.0, 0.06040243, 0.54321383, 0.0, 0.0],
        ]
    )
    print(W1.sum(0))
    print(W1.sum(0))
    #
    assert np.allclose(W1, W1.T) and np.allclose(W2, W2.T)
    # assert np.allclose(W1.sum(0), 1) and np.allclose(W1.sum(1), 1)
    # assert np.allclose(W2.sum(0), 1) and np.allclose(W2.sum(1), 1)
    assert np.all(np.diag(W1) == 0) and np.all(np.diag(W2) == 0)
    assert np.all(W2[W1 == 0.0] == 0.0)

    if not isinstance(W1, np.ndarray) or not isinstance(W2, np.ndarray):
        raise ValueError("Inputs must be NumPy arrays.")
    if W1.shape != W2.shape or len(W1.shape) != 2 or W1.shape[0] != W1.shape[1]:
        raise ValueError("Inputs must be square matrices of the same size.")

    # Create a custom light colormap for normal values
    light_cmap = ListedColormap(["#f8f8ff", "#d3d3d3", "#b0c4de", "#add8e6", "#87ceeb", "#4682b4"])

    # Prepare masks for elements below the threshold
    mask1 = W1 < threshold
    masked_W1 = np.ma.masked_where(mask1, W1)

    mask2 = W2 < threshold
    masked_W2 = np.ma.masked_where(mask2, W2)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))  # Side-by-side heatmaps
    ax1, ax2 = axes

    # Plot for W1
    ax1.imshow(masked_W1, cmap=light_cmap, interpolation="nearest")
    ax1.imshow(mask1, cmap=ListedColormap(["black"]), interpolation="nearest", alpha=mask1 * 1.0)
    ax1.set_aspect(0.5)  # Rectangular cells (length = 2 * width)
    ax1.axis("off")
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            value = W1[i, j]
            if value < threshold:
                ax1.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
            else:
                ax1.text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=8)
    ax1.text(4, 11, "$W^r$", fontsize=20, weight="bold")

    # Plot for W2
    ax2.imshow(masked_W2, cmap=light_cmap, interpolation="nearest")
    ax2.imshow(mask2, cmap=ListedColormap(["black"]), interpolation="nearest", alpha=mask2 * 1.0)
    ax2.set_aspect(0.5)  # Rectangular cells (length = 2 * width)
    ax2.axis("off")
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            value = W2[i, j]
            if value < threshold:
                ax2.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
            else:
                ax2.text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=8)
    ax2.text(4, 11, "$W^r_p$", fontsize=20, weight="bold", horizontalalignment="left", verticalalignment="bottom")

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_train_time(logs_time, metric="accuracy", measure="mean", info=None, plot_peer=None):
    logs, times = logs_time
    if isinstance(logs, str):
        logs = load(logs)
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    # prepare data
    std_data = None
    if measure == "mean":
        data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        print(data)
    elif measure == "mean-std":
        if plot_peer is None:
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            print(f">>>>> Plotting chart for Peer({plot_peer})...")
            data = [v[metric] for v in logs[plot_peer]]
        std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    elif measure == "max":
        data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    else:
        data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    # plot data
    xlabel = "Rounds"
    ylabel = f" {measure.capitalize()} {_metric.capitalize()}"
    # title = f'{_metric.capitalize()} vs. No. of rounds'
    title = None
    if info:
        xlabel = info.get("xlabel", xlabel)
        ylabel = info.get("ylabel", ylabel)
        title = info.get("title", title)
    # x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    x = times
    # Configs
    plt.grid(linestyle="dashed")
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(
        fontsize=13,
    )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(
        fontsize=13,
    )
    # Plot
    plt.plot(x, data)
    plt.scatter(x, data, color="red")
    if std_data is not None:
        plt.fill_between(x, data - std_data, data + std_data, alpha=0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=9)

    plt.show()


def plot_train_history(train, times, metric="accuracy", measure="mean", info=None, plot_peer=None):
    if isinstance(train, str):
        train = load(train)
    # Get metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)

    # Prepare data
    std_data = None
    if measure == "mean":
        data = np.mean([[v[metric] for v in lo] for lo in train.values()], axis=0)
    elif measure == "mean-std":
        if plot_peer is None:
            data = np.mean([[v[metric] for v in lo] for lo in train.values()], axis=0)
        else:
            print(f">>>>> Plotting chart for Peer({plot_peer})...")
            data = [v[metric] for v in train[plot_peer]]
        std_data = np.std([[v[metric] for v in lo] for lo in train.values()], axis=0)
    elif measure == "max":
        data = np.max([[v[metric] for v in lo] for lo in train.values()], axis=0)
        std_data = np.std([[v[metric] for v in lo] for lo in train.values()], axis=0)
    else:
        data = np.std([[v[metric] for v in lo] for lo in train.values()], axis=0)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Common settings
    x_rounds = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    ylabel = f"{measure.capitalize()} {_metric.capitalize()}"
    title = info.get("title", None) if info else None

    # Plot accuracy vs rounds
    ax1.plot(x_rounds, data)
    if std_data is not None:
        ax1.fill_between(x_rounds, data - std_data, data + std_data, alpha=0.1)
    ax1.set_xlabel(info.get("xlabel", "Rounds") if info else "Rounds", fontsize=13)
    ax1.set_ylabel(ylabel, fontsize=13)
    ax1.grid(linestyle="dashed")

    # Plot accuracy vs time
    ax2.plot(times, data)
    if std_data is not None:
        ax2.fill_between(times, data - std_data, data + std_data, alpha=0.1)
    ax2.set_xlabel(info.get("time_xlabel", "Cumulative Time") if info else "Cumulative Time", fontsize=13)
    ax2.grid(linestyle="dashed")

    # Formatting
    for ax in [ax1, ax2]:
        ax.tick_params(axis="both", labelsize=13)
        if title:
            ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_train_history_old(train, times, metric="accuracy", measure="mean", info=None, plot_peer=None):
    if isinstance(train, str):
        train = load(train)
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    # prepare data
    std_data = None
    if measure == "mean":
        data = np.mean([[v[metric] for v in lo] for lo in train.values()], axis=0)
    elif measure == "mean-std":
        if plot_peer is None:
            data = np.mean([[v[metric] for v in lo] for lo in train.values()], axis=0)
        else:
            print(f">>>>> Plotting chart for Peer({plot_peer})...")
            data = [v[metric] for v in train[plot_peer]]
        std_data = np.std([[v[metric] for v in lo] for lo in train.values()], axis=0)
    elif measure == "max":
        data = np.max([[v[metric] for v in lo] for lo in train.values()], axis=0)
    else:
        data = np.std([[v[metric] for v in lo] for lo in train.values()], axis=0)
    # plot data
    xlabel = "Rounds"
    ylabel = f" {measure.capitalize()} {_metric.capitalize()}"
    # title = f'{_metric.capitalize()} vs. No. of rounds'
    title = None
    if info:
        xlabel = info.get("xlabel", xlabel)
        ylabel = info.get("ylabel", ylabel)
        title = info.get("title", title)
    x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    # Configs
    plt.grid(linestyle="dashed")
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(
        fontsize=13,
    )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(
        fontsize=13,
    )
    plt.ylim(0, 1.05)
    # Plot
    plt.plot(x, data)
    if std_data is not None:
        plt.fill_between(x, data - std_data, data + std_data, alpha=0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=9)

    plt.show()


"""
    X ---> 100
    a ---> 100-86
"""


def plot_trans_energy(dataset="mnist"):
    if dataset == "mnist":
        data_W = load("MNIST_PERF_ROUNDS_251_BASE2.pkl")
        data_WP = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
        data_WP_G1 = load("MNIST_PERF_ROUNDS_251_G1.pkl")
    else:
        data_W = load("CIFAR_PERF_ROUNDS_251_BASE.pkl")
        data_WP = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")
        data_WP_G1 = load("CIFAR_PERF_ROUNDS_251_G1.pkl")

    base = np.sum([vals[2] for vals in data_W["energy"].values()]) / 251 * 100
    wp_g1 = np.sum([vals[3] for vals in data_WP_G1["energy"].values()]) / 251 * 90
    wp_gopt = np.sum([vals[3] for vals in data_WP["energy"].values()]) / 251 * 100
    trans_energy = {  # mnist old
        "$\mathbf{W}, \gamma=1$": base,
        "$\mathbf{W_p}, \gamma=1$": wp_g1,
        "$\mathbf{W_p}, \gamma_{OPT}$": wp_gopt,
    }
    print(trans_energy)

    labels = list(trans_energy.keys())
    values = list(trans_energy.values())
    plt.figure()
    colors = ["skyblue", "skyblue", "lightgreen", "lightgreen", "skyblue", "red"]
    # colors = ['skyblue'] * 7
    if dataset == "mnist":
        vals = ["100\nRounds", "70\nRounds", "120\nRounds"]
    else:
        vals = ["100\nRounds", "90\nRounds", "110\nRounds"]
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    has_text = False
    if has_text:
        for bar, value in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # Center of the bar
                bar.get_height() / 2,  # Vertical position inside the bar
                f"{value}",  # Format the value to 4 decimal places
                ha="center",  # Horizontal alignment
                va="center",  # Vertical alignment
                color="black",  # Text color
                fontsize=10,
            )

    # Adding title and labels
    plt.grid(linestyle="dashed")
    plt.xlabel("Network configurations", fontsize=14)
    plt.ylabel("Transmission Energy [J]", fontsize=13)
    # Customize x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=13)
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Show the plot
    plt.show()


def plot_compute_energy(dataset="mnist"):
    if dataset == "mnist":
        data = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
    else:
        data = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")

    base = np.sum([vals[0] for vals in data["energy"].values()]) * 80 * 100 / 251  # * 100 / 251  #
    wp_fmax = np.sum([vals[0] for vals in data["energy"].values()]) * 80 * 90 / 251  # * 70 / 251  #
    wp_fopt = np.sum([vals[1] for vals in data["energy"].values()]) * 80 * 3.6  #

    compute_energy = {r"$\boldsymbol{W}, \boldsymbol{f}_{2GHz}$": base, r"$\boldsymbol{W_p}, \boldsymbol{f}_{2GHz}$": wp_fmax, r"$\boldsymbol{W_p}, \boldsymbol{f}_{OPT}$": wp_fopt * 10}
    print(compute_energy)
    labels = list(compute_energy.keys())
    values = list(compute_energy.values())
    plt.figure()
    colors = ["skyblue", "skyblue", "lightgreen", "skyblue", "skyblue", "skyblue", "red"]
    # colors = ['skyblue'] * 7
    if dataset == "mnist":
        vals = ["100\nRounds", "70\nRounds", "120\nRounds"]
    else:
        vals = ["100\nRounds", "90\nRounds", "110\nRounds"]
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    # for bar, value in zip(bars, vals):
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2,  # Center of the bar
    #         300 if value == '110\nRounds' else bar.get_height() / 2,  # Vertical position inside the bar
    #         f"{value}",  # Format the value to 4 decimal places
    #         ha='center',  # Horizontal alignment
    #         va='center',  # Vertical alignment
    #         color='black',  # Text color
    #         fontsize=10
    #     )

    # Adding title and labels
    plt.grid(linestyle="dashed")
    plt.xlabel("CPU Frequency", fontsize=14)
    plt.ylabel("Computation Energy [J]", fontsize=13)
    # Customize x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=13)
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Show the plot
    plt.show()


def plot_compu2_energy():
    # total_energy = { # mnist
    #     "$W, \gamma=1$": 0.10686077166677443,
    #     "$W_p, \gamma=1$": 0.0676431849027338,
    #     "$W_p, \gamma=0.5$": 0.05357876031899707,
    #     "$W_p, \gamma=0.4$": 0.05893663635089678,
    #     "$W_p, \gamma=0.3$": 0.07634973345457081,
    #     "$W_p, \gamma=0.2$": 0.06697345039874635,
    #     "$W_p, \gamma=0.1$": 0.03348672519937317,
    # }
    total_energy = {  # cifar
        "$W, \gamma=1$": 9.86,
        "$W_p, \gamma=1$": 8.56,
        "$W_p, \gamma=0.5$": 6.88,
        "$W_p, \gamma=0.4$": 6.18,
        "$W_p, \gamma=0.3$": 9.27,
        "$W_p, \gamma=0.2$": 7.73,
        "$W_p, \gamma=0.1$": 4.3080,
    }
    labels = list(total_energy.keys())
    values = list(total_energy.values())
    plt.figure()
    colors = ["lightgreen", "skyblue", "skyblue", "skyblue", "skyblue", "skyblue", "red"]
    # vals = ['100\nRounds', '100\nRounds', '160\nRounds', '220\nRounds', '380\nRounds', '500\nRounds', '500 \nRounds']
    vals = ["90\nRounds", "100\nRounds", "160\nRounds", "180\nRounds", "360\nRounds", "450\nRounds", "500 \nRounds"]
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    for bar, value in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height() / 2,  # Vertical position inside the bar
            f"{value}",  # Format the value to 4 decimal places
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            color="black",  # Text color
            fontsize=10,
        )

    # Adding title and labels
    plt.grid(linestyle="dashed")
    plt.xlabel("Configurations", fontsize=14)
    plt.ylabel("Transmission Energy [J]", fontsize=13)
    # Customize x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=13)
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Show the plot
    plt.show()


def plot_rounds(dataset="mnist", metric="accuracy", measure="max", info=None, use_time=False):
    if dataset == "mnist":
        gamma00 = load("MNIST_PERF_ROUNDS_251_BASE.pkl")
        gamma10 = load("MNIST_PERF_ROUNDS_251_G1.pkl")
        gamma99 = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
        # gamma01 = load("mnist/MNIST_G_0.1_501.pkl")
        # gamma02 = load("mnist/MNIST_G_0.2_501.pkl")
        # gamma03 = load("mnist/MNIST_G_0.3_501.pkl")
        # gamma04 = load("mnist/MNIST_G_0.4_501.pkl")
        # gamma05 = load("mnist/MNIST_G_0.5_501.pkl")
    else:
        gamma00 = load("CIFAR_PERF_ROUNDS_251_BASE.pkl")
        gamma10 = load("CIFAR_PERF_ROUNDS_251_G1.pkl")
        # gamma99 = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")
        # gamma01 = load("CIFAR_PERF_ROUNDS_OPT_501.pkl")
        # gamma02 = load("cifar/CIFAR_G_0.2_251.pkl")
        # gamma03 = load("cifar/CIFAR_G_0.3_251.pkl")
        # gamma04 = load("cifar/CIFAR_G_0.4_251.pkl")
        # gamma05 = load("cifar/CIFAR_G_0.5_251.pkl")

    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    # data00
    if dataset == "mnist":
        limit_rounds = 21
        limit_acc = None
    else:
        limit_rounds = 25
        limit_acc = None
    data00 = np.max([[v[metric] for v in lo] for lo in gamma00["train"].values()], axis=0)[:limit_rounds]
    data00 = gaussian_filter1d(data00, sigma=1)
    std_data00 = np.std([[v[metric] for v in lo] for lo in gamma00["train"].values()], axis=0)
    lnd00 = len(data00)
    lnd00 = (
        np.argmax(np.array(data00) >= limit_acc) if limit_acc and max(np.array(data00)) >= limit_acc else len(data00)
    )
    x00 = range(0, lnd00 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma00["time"][:lnd00]
    # data10
    data10 = np.max([[v[metric] for v in lo] for lo in gamma10["train"].values()], axis=0)[:limit_rounds]
    data10 = gaussian_filter1d(data10, sigma=1)
    std_data10 = np.std([[v[metric] for v in lo] for lo in gamma10["train"].values()], axis=0)
    lnd10 = len(data10)
    lnd10 = (
        np.argmax(np.array(data10) >= limit_acc) if limit_acc and max(np.array(data10)) >= limit_acc else len(data10)
    )
    x10 = range(0, lnd10 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma10["time"][:lnd10]
    # data99
    # data99 = np.max([[v[metric] for v in lo] for lo in gamma99["train"].values()], axis=0)[:limit_rounds]
    # data99 = gaussian_filter1d(data99, sigma=1)
    # std_data99 = np.std([[v[metric] for v in lo] for lo in gamma99["train"].values()], axis=0)
    # lnd99 = len(data99)  # MNIST 16 | CIFAR 170 # len(data05)
    # lnd99 = (
    #     np.argmax(np.array(data99) >= limit_acc) if limit_acc and max(np.array(data99)) >= limit_acc else len(data99)
    # )
    # x99 = range(0, lnd99 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma99["time"][:lnd99]
    # # data01
    # data01 = np.mean([[v[metric] for v in lo] for lo in gamma01["train"].values()], axis=0)[:limit_rounds]
    # data01 = gaussian_filter1d(data01, sigma=1)
    # std_data01 = np.std([[v[metric] for v in lo] for lo in gamma01["train"].values()], axis=0)
    # lnd01 = len(data01)  # MNIST 50 | CIFAR  # len(data01)
    # x01 = range(0, lnd01 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma01["time"][:21]
    # # data02
    # data02 = np.mean([[v[metric] for v in lo] for lo in gamma02["train"].values()], axis=0)[:limit_rounds]
    # data02 = gaussian_filter1d(data02, sigma=1)
    # std_data02 = np.std([[v[metric] for v in lo] for lo in gamma02["train"].values()], axis=0)
    # lnd02 = len(data02)  # MNIST 50 | CIFAR 501 # len(data02)
    # x02 = range(0, lnd02 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma02["time"][:21]
    # # data03
    # data03 = np.mean([[v[metric] for v in lo] for lo in gamma03["train"].values()], axis=0)[:limit_rounds]
    # data03 = gaussian_filter1d(data03, sigma=1)
    # std_data03 = np.std([[v[metric] for v in lo] for lo in gamma03["train"].values()], axis=0)
    # lnd03 = len(data03)  # MNIST 38 | CIFAR 360 # len(data03)
    # x03 = range(0, lnd03 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma03["time"][:21]
    # # data04
    # data04 = np.mean([[v[metric] for v in lo] for lo in gamma04["train"].values()], axis=0)[:limit_rounds]
    # data04 = gaussian_filter1d(data04, sigma=1)
    # std_data04 = np.std([[v[metric] for v in lo] for lo in gamma04["train"].values()], axis=0)
    # lnd04 = len(data04)  # MNIST 22 | CIFAR 190 # len(data04)
    # x04 = range(0, lnd04 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma04["time"][:21]
    # # data05
    # data05 = np.mean([[v[metric] for v in lo] for lo in gamma05["train"].values()], axis=0)[:limit_rounds]
    # data05 = gaussian_filter1d(data05, sigma=1)
    # std_data05 = np.std([[v[metric] for v in lo] for lo in gamma05["train"].values()], axis=0)
    # lnd05 = len(data05)  # MNIST 16 | CIFAR 170 # len(data05)
    # x05 = range(0, lnd05 * EVAL_ROUND, EVAL_ROUND) if not use_time else gamma05["time"][:21]

    # plot data
    xlabel = "Execution Time (seconds)" if use_time else "Number of rounds"
    ylabel = "Test Accuracy"
    title = None
    if info:
        xlabel = info.get("xlabel", xlabel)
        ylabel = info.get("ylabel", ylabel)
        title = info.get("title", title)
    # , color=colors[i], label=mean[i][1], linestyle=line_styles[i]
    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    # line_styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    # , marker=markers[0], markersize=4, linewidth=3
    colors = ["#000", "#2ca02c", "#bcbd22", "#17becf", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#ff7f0e"]
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X"]

    if limit_acc:
        plt.axhline(y=limit_acc, color="red", linestyle="--", alpha=0.3)
    plt.plot(x00, data00[:lnd00], label="$\mathbf{W}$", color=colors[0], linewidth=2)
    plt.fill_between(x00, data00[:lnd00] - std_data00[:lnd00], data00[:lnd00] + std_data00[:lnd00], alpha=0.1)

    # plt.plot(x99, data99[:lnd99], label="$W_p, \gamma=OPT$", color=colors[2], marker=markers[1], markersize=2)
    # plt.fill_between(x99, data99[:lnd99] - std_data99[:lnd99], data99[:lnd99] + std_data99[:lnd99], alpha=0.1)

    plt.plot(x10, data10[:lnd10], label="$\mathbf{W_p}$", color=colors[1], marker=markers[6], markersize=2)
    plt.fill_between(x10, data10[:lnd10] - std_data10[:lnd10], data10[:lnd10] + std_data10[:lnd10], alpha=0.1)

    # plt.plot(x01, data01[:lnd01], label="$W_p, \gamma=0.1$", color=colors[3], marker=markers[2], markersize=2)
    # plt.fill_between(x01, data01[:lnd01] - std_data01[:lnd01], data01[:lnd01] + std_data01[:lnd01], alpha=.1)
    #
    # plt.plot(x02, data02[:lnd02], label="$W_p, \gamma=0.2$", color=colors[4], marker=markers[3], markersize=2)
    # plt.fill_between(x02, data02[:lnd02] - std_data02[:lnd02], data02[:lnd02] + std_data02[:lnd02], alpha=.1)

    # plt.plot(x03, data03[:lnd03], label="$W_p, \gamma=0.3$", color=colors[5], marker=markers[4], markersize=2)
    # plt.fill_between(x03, data03[:lnd03] - std_data03[:lnd03], data03[:lnd03] + std_data03[:lnd03], alpha=.1)

    # plt.plot(x04, data04[:lnd04], label="$W_p, \gamma=0.4$", color=colors[6], marker=markers[5], markersize=2)
    # plt.fill_between(x04, data04[:lnd04] - std_data04[:lnd04], data04[:lnd04] + std_data04[:lnd04], alpha=.1)
    #
    # plt.plot(x05, data05[:lnd05], label="$W_p, \gamma=0.5$", color=colors[7], marker=markers[6], markersize=2)
    # plt.fill_between(x05, data05[:lnd05] - std_data05[:lnd05], data05[:lnd05] + std_data05[:lnd05], alpha=.1)

    # Configs
    plt.grid(linestyle="dashed")
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(
        fontsize=13,
    )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(
        fontsize=13,
    )
    if title is not None:
        plt.title(title, fontsize=9)
    plt.legend(loc="lower right", markerscale=2, fontsize=14)
    plt.tight_layout()
    plt.show()


def process_data(data, metric, use_time):
    if data is None:
        return None, None, None, 0

    try:
        metric_values = [[v[metric] for v in lo] for lo in data["train"].values()]
        mean_data = np.mean(metric_values, axis=0)
        mean_data_filtered = gaussian_filter1d(mean_data, sigma=1)
        std_data = np.std(metric_values, axis=0)

        reached_threshold_indices = np.where(np.array(mean_data_filtered) >= 0.9)[0]
        lnd = reached_threshold_indices[0] + 1 if reached_threshold_indices.size > 0 else len(mean_data_filtered)

        if use_time and "time" in data:
            x_values = data["time"][:lnd]
        else:
            x_values = range(0, lnd * EVAL_ROUND, EVAL_ROUND)[:lnd]
            if use_time and "time" not in data:
                print("Warning: 'time' key not found in data, using rounds for x-axis.")

        return mean_data_filtered[:lnd], std_data[:lnd], x_values, lnd
    except (KeyError, Exception) as e:
        print(f"Error processing data: {e}")
        return None, None, None, 0


def plot_time(dataset="mnist", metric="accuracy", measure="mean", info=None, use_time=True, target_acc=None):
    try:
        if dataset == "mnist":
            gamma00 = load("MNIST_PERF_ROUNDS_251_BASE.pkl")
            gamma10 = load("MNIST_PERF_ROUNDS_251_G1.pkl")
            gamma99 = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
        else:
            gamma00 = load("CIFAR_PERF_ROUNDS_251_BASE.pkl")
            gamma10 = load("CIFAR_PERF_ROUNDS_251_G1.pkl")
            gamma99 = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")

    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    metric, measure = verify_metrics(metric, measure)

    # Modified process_data call to include target_acc
    def process_with_target(data, metric, use_time):
        if data is None:
            return None, None, None, 0
        try:
            metric_values = [[v[metric] for v in lo] for lo in data["train"].values()]
            mean_data = np.mean(metric_values, axis=0)
            mean_data_filtered = gaussian_filter1d(mean_data, sigma=1)
            std_data = np.std(metric_values, axis=0)

            # Use target_acc if provided, otherwise default to 0.9
            threshold = target_acc if target_acc is not None else 0.9
            reached_threshold_indices = np.where(np.array(mean_data_filtered) >= threshold)[0]
            lnd = reached_threshold_indices[0] + 1 if reached_threshold_indices.size > 0 else len(mean_data_filtered)

            if use_time and "time" in data:
                x_values = data["time"][:lnd]
            else:
                x_values = list(range(0, lnd * EVAL_ROUND, EVAL_ROUND))
                x_values = x_values[:lnd]
                if use_time and "time" not in data:
                    print("Warning: 'time' key not found in data, using rounds for x-axis.")

            return mean_data_filtered[:lnd], std_data[:lnd], x_values, lnd
        except (KeyError, Exception) as e:
            print(f"Error processing data: {e}")
            return None, None, None, 0

    data00, std_data00, x00, lnd00 = process_with_target(gamma00, metric, use_time)
    data10, std_data10, x10, lnd10 = process_with_target(gamma10, metric, use_time)
    data99, std_data99, x99, lnd99 = process_with_target(gamma99, metric, use_time)

    if any(d is None for d in [data00, data10, data99]):
        print("Skipping plot due to data loading or processing errors.")
        return

    xlabel = "Execution Time (seconds)" if use_time else "Number of rounds"
    ylabel = "Test Accuracy"
    title = None
    if info:
        xlabel = info.get("xlabel", xlabel)
        ylabel = info.get("ylabel", ylabel)
        title = info.get("title", title)

    colors = ["#000", "#2ca02c", "#bcbd22"]
    markers = ["o", "s", "D"]

    plt.figure()
    if dataset == "mnist":
        cifar_time = 1
    else:
        cifar_time = 8.5
    plt.plot(x00 * cifar_time, data00, label="$\mathbf{W}$", color=colors[0], linewidth=2)
    plt.fill_between(x00 * cifar_time, data00 - std_data00, data00 + std_data00, alpha=0.1)

    plt.plot(x99 * cifar_time, data99, label="$\mathbf{W_p}, \gamma_{OPT}$", color=colors[2], marker=markers[1], markersize=2)
    plt.fill_between(x99 * cifar_time, data99 - std_data99, data99 + std_data99, alpha=0.1)

    plt.plot(x10 * cifar_time, data10, label="$\mathbf{W_p}, \gamma=1$", color=colors[1], marker=markers[2], markersize=2)
    plt.fill_between(x10 * cifar_time, data10 - std_data10, data10 + std_data10, alpha=0.1)

    # Add horizontal line for target accuracy if specified
    if target_acc is not None:
        plt.axhline(y=target_acc, color="red", linestyle="--", alpha=0.3)

    plt.grid(linestyle="dashed")
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13)
    if title is not None:
        plt.title(title, fontsize=9)
    plt.legend(loc="lower right", markerscale=2, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_manymore(exps, metric="accuracy", measure="mean", info=None, save=False):
    # Configs
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    xlabel = "Rounds"
    ylabel = "Test Accuracy"
    title = f"{_metric.capitalize()} vs. No. of rounds"
    if info is not None:
        xlabel = info.get("xlabel", xlabel)
        ylabel = info.get("ylabel", ylabel)
        title = info.get("title", title)
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel(xlabel, fontsize=13)
    colors = ["green", "blue", "orange", "black", "red", "grey", "tan", "pink", "navy", "aqua"]
    # colors = ['black', 'green', 'orange', 'blue', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']
    line_styles = ["-", "--", "-.", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    plt.grid(linestyle="dashed")
    plt.rc("legend", fontsize=12)
    plt.xticks(
        fontsize=13,
    )
    plt.yticks(
        fontsize=13,
    )
    std_data = None
    for i, exp in enumerate(exps):
        # Data
        logs = load(exp["file"])
        name = exp.get("name", "")
        if measure == "mean":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "mean-std":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
            std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "max":
            data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
        plt.plot(x, data, color=colors[i], label=name, linestyle=line_styles[i])
        if std_data is not None:
            plt.fill_between(x, data - std_data, data + std_data, color=colors[i], alpha=0.1)

    plt.legend(loc="lower right", shadow=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title(title)
    if save:
        unique = np.random.randint(100, 999)
        plt.savefig(f"../out/EXP_{unique}.pdf")
    plt.show()


# ===***===***===***===***===***===***===***===***===***===***===***===***===***===***===***===***


def plot_w_wp(metric="accuracy", measure="mean", info=None):
    no_op = load("dfl_log_10_101_opt_False_gamma_1_250.pkl")["train"]
    ye_op = load("dfl_log_10_101_opt_True_gamma_1_250.pkl")["train"]
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    no_data = np.mean([[v[metric] for v in lo] for lo in no_op.values()], axis=0)
    ye_data = np.mean([[v[metric] for v in lo] for lo in ye_op.values()], axis=0)
    noo_std = np.std([[v[metric] for v in lo] for lo in no_op.values()], axis=0)
    yes_std = np.std([[v[metric] for v in lo] for lo in ye_op.values()], axis=0)
    # plot data
    xlabel = "Number of rounds"
    ylabel = "Test Accuracy"
    title = None
    if info:
        xlabel = info.get("xlabel", xlabel)
        ylabel = info.get("ylabel", ylabel)
        title = info.get("title", title)
    x = range(0, len(no_data) * EVAL_ROUND, EVAL_ROUND)
    # , color=colors[i], label=mean[i][1], linestyle=line_styles[i]
    plt.plot(x, no_data, label="Without network optimization", linestyle="-.")  # , '-x'
    plt.fill_between(x, no_data - noo_std, no_data + noo_std, alpha=0.1)
    plt.plot(x, ye_data, label="With network optimization", linestyle="-")
    plt.fill_between(x, ye_data - yes_std, ye_data + yes_std, alpha=0.1)
    # Configs
    plt.grid(linestyle="dashed")
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(
        fontsize=13,
    )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(
        fontsize=13,
    )
    if title is not None:
        plt.title(title, fontsize=9)
    plt.legend()
    plt.show()


def plot_accuracy_vs_time():
    # Calculate cumulative times and average accuracies
    # MNIST
    # raw_times = load("time_data_0.1_560.pkl")
    # opt_times = load("cifar_time_data_opt.pkl")
    # raw_times = load("FAR_TIME_W_501_550.pkl")
    # opt_times = load("CIFAR_TIME_WP_501_386.pkl")
    raw_times = load("MNIST_MER_W_501_377.pkl")
    opt_times = load("MNIST_MER_WP_501_608.pkl")
    # CIFAR
    # raw_times = load("time_data_0.1_196.pkl")
    # nof_times = load("time_data_0.1_156.pkl")
    # opt_times = load("time_data_0.1_740.pkl")
    # Calculate cumulative times and average accuracies per round
    cumulative_raw = []
    cumulative_opt = []
    cumulative_nof = []
    raw_accuracies = []
    opt_accuracies = []
    nof_accuracies = []
    current_cum_raw = 0.0
    current_cum_opt = 0.0
    current_cum_nof = 0.0

    # Get number of rounds from first peer's data
    num_rounds = len(opt_times[0])
    # num_rounds = 16

    for round_idx in range(num_rounds):
        max_raw = max(peer[round_idx]["raw_time"] for peer in raw_times)
        max_opt = max(peer[round_idx]["opt_time"] for peer in opt_times) / 2
        max_nof = max(peer[round_idx]["nof_time"] for peer in opt_times)

        # Update cumulative times
        current_cum_raw += max_raw
        current_cum_opt += max_opt
        current_cum_nof += max_nof
        cumulative_raw.append(current_cum_raw)
        cumulative_opt.append(current_cum_opt)
        cumulative_nof.append(current_cum_nof)
        print(f"Round {round_idx} >> RAW: {current_cum_raw} | OPT: {current_cum_opt} | NOF: {current_cum_nof}")

        # Calculate average accuracy for this round
        avg_acc = max(peer[round_idx]["val_acc"] for peer in opt_times)
        raw_accuracies.append(avg_acc)
        avg_acc = max(peer[round_idx]["val_acc"] for peer in opt_times)
        opt_accuracies.append(avg_acc)
        avg_acc = max(peer[round_idx]["val_acc"] for peer in opt_times)
        nof_accuracies.append(avg_acc)

    # Create plot
    plt.figure()

    # Plot both time curves
    plt.plot(
        cumulative_raw, raw_accuracies, marker="o", markersize=3, linestyle="-", linewidth=2, color="black", label="$W$"
    )

    plt.plot(
        cumulative_opt,
        opt_accuracies,
        marker="s",
        markersize=3,
        linestyle="-",
        linewidth=2,
        color="orange",
        label="$W_p, f_{OPT}$",
    )

    plt.plot(
        cumulative_nof,
        nof_accuracies,
        marker=">",
        markersize=3,
        linestyle="-",
        linewidth=2,
        color="blue",
        label="$W_p, f_{max}$",
    )

    xlabel = "Execution Time (seconds)"
    ylabel = "Test Accuracy"
    # plt.title('Validation Accuracy vs Cumulative Training Time (All Rounds)')
    plt.ylim(0, 1.05)
    # Configs
    plt.grid(linestyle="dashed")
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(
        fontsize=13,
    )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(
        fontsize=13,
    )
    plt.legend(markerscale=2, loc="lower right")

    # Add round number labels every 5 rounds
    for i in range(0, num_rounds, 10):
        plt.annotate(
            f"R{i * 10}",
            (cumulative_raw[i], raw_accuracies[i]),
            textcoords="offset points",
            xytext=(0, -20),
            ha="right",
            color="black",
        )

        plt.annotate(
            f"R{i * 10}",
            (cumulative_nof[i], nof_accuracies[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            color="blue",
        )

        plt.annotate(
            f"R{i * 10}",
            (cumulative_opt[i], opt_accuracies[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            color="orange",
        )

    plt.tight_layout()
    plt.show()


def plot_freq_time(dataset="mnist", target_accuracy=0.94, ylim=1.0):
    # Load data
    # MNIST
    if dataset == "mnist":
        raw_times = load("MNIST_PERF_ROUNDS_251_G1_DELTA.pkl")["time_delta"]
        opt_times = load("MNIST_PERF_ROUNDS_251_OPT_DELTA.pkl")["time_delta"]
    else:
        raw_times = load("CIFAR_PERF_ROUNDS_251_G1_DELTA.pkl")["time_delta"]
        opt_times = load("CIFAR_PERF_ROUNDS_251_G1_DELTA.pkl")["time_delta"]

    for round_idx in range(len(opt_times[0])):
        print(
            f"Round {round_idx} >> RAW: {opt_times[0][round_idx]['raw_time']} | OPT: "
            f"{opt_times[0][round_idx]['opt_time']} | NOF: {opt_times[0][round_idx]['nof_time']}"
        )

    # Initialize cumulative variables
    cumulative_raw, cumulative_opt, cumulative_nof = [], [], []
    raw_accuracies, opt_accuracies, nof_accuracies = [], [], []
    current_cum_raw = current_cum_opt = current_cum_nof = 0.0

    # Get number of rounds from first peer's data
    num_rounds = len(opt_times[0])

    # Track if we've reached target for each method
    raw_done = opt_done = nof_done = False

    for round_idx in range(num_rounds):
        # Calculate max times for this round
        max_raw = max(peer[round_idx]["raw_time"] for peer in raw_times)
        max_opt = max(peer[round_idx]["opt_time"] for peer in opt_times)
        max_nof = max(peer[round_idx]["nof_time"] for peer in opt_times)

        # Update cumulative times
        current_cum_raw += max_raw
        current_cum_opt += max_opt
        current_cum_nof += max_nof

        # Calculate average accuracies
        avg_raw = max(peer[round_idx]["val_acc"] for peer in raw_times)
        avg_opt = max(peer[round_idx]["val_acc"] for peer in opt_times)
        avg_nof = max(peer[round_idx]["val_acc"] for peer in opt_times)

        # Only append if we haven't reached target yet
        if not raw_done:
            cumulative_raw.append(current_cum_raw)
            raw_accuracies.append(avg_raw)
            if target_accuracy is not None and avg_raw >= target_accuracy:
                raw_done = True

        if not opt_done:
            cumulative_opt.append(current_cum_opt)
            opt_accuracies.append(avg_opt)
            if target_accuracy is not None and avg_opt >= target_accuracy:
                opt_done = True

        if not nof_done:
            cumulative_nof.append(current_cum_nof)
            nof_accuracies.append(avg_nof)
            if target_accuracy is not None and avg_nof >= target_accuracy:
                nof_done = True

        # Early exit if all methods have reached target
        if raw_done and opt_done and nof_done:
            break

    # Create plot
    plt.figure()

    # Plot curves (only if we have data points)
    if len(cumulative_raw) > 0:
        plt.plot(
            cumulative_raw,
            raw_accuracies,
            marker="o",
            markersize=3,
            linestyle="-",
            linewidth=2,
            color="black",
            label="$W, f_{max}$",
        )

    if len(cumulative_opt) > 0:
        plt.plot(
            cumulative_opt,
            opt_accuracies,
            marker="s",
            markersize=3,
            linestyle="-",
            linewidth=2,
            color="orange",
            label="$W_p, f_{OPT}$",
        )

    if len(cumulative_nof) > 0:
        plt.plot(
            cumulative_nof,
            nof_accuracies,
            marker="d",
            markersize=3,
            linestyle="-",
            linewidth=2,
            color="blue",
            label="$W_p, f_{max}$",
        )

    # Add a horizontal line at target accuracy
    if target_accuracy is not None:
        plt.axhline(y=target_accuracy, color="red", linestyle="--", alpha=0.3)

    # Plot configuration
    plt.xlabel("Execution Time (seconds)", fontsize=13)
    plt.ylabel("Test Accuracy", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim(0, ylim)
    plt.grid(linestyle="dashed")
    plt.legend(loc="lower right", markerscale=2, fontsize=14)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_energy_comparison(dataset="mnist"):
    # Prepare data
    configurations = ["W", "WP", "WP γ=1", "WP f=max"]

    if dataset == "mnist":
        data_W = load("MNIST_PERF_ROUNDS_501_BASE.pkl")
        data_WP = load("MNIST_PERF_ROUNDS_501_OPT.pkl")
        data_WP_G1 = load("MNIST_PERF_ROUNDS_501_G1.pkl")
        data_WP_FMAX = load("MNIST_MER_OPT_501_822.pkl")
    else:
        data_W = load("MNIST_MER_OPT_501_822.pkl")
        data_WP = load("MNIST_MER_OPT_501_822.pkl")
        data_WP_G1 = load("MNIST_MER_OPT_501_822.pkl")
        data_WP_FMAX = load("MNIST_MER_OPT_501_822.pkl")
    x = data_W["energy"]

    # Calculate total energy for each configuration (with and without optimization)
    # Without optimization (cp_energy + ts_energy)
    totals_no_opt = [
        np.sum([vals[0] + vals[2] for vals in data_W["energy"].values()]),
        np.sum([vals[0] + vals[2] for vals in data_WP["energy"].values()]),
        np.sum([vals[0] + vals[2] for vals in data_WP_G1["energy"].values()]),
        np.sum([vals[0] + vals[2] for vals in data_WP_FMAX["energy"].values()]),
    ]

    # With optimization (cp_energy_opt + ts_energy_opt)
    totals_with_opt = [
        np.sum([vals[1] + vals[3] for vals in data_W["energy"].values()]),
        np.sum([vals[1] + vals[3] for vals in data_WP["energy"].values()]),
        np.sum([vals[1] + vals[3] for vals in data_WP_G1["energy"].values()]),
        np.sum([vals[1] + vals[3] for vals in data_WP_FMAX["energy"].values()]),  # Assuming same as above for WP,f=max
    ]

    # Plot settings
    bar_width = 0.35
    index = np.arange(len(configurations))

    plt.figure(figsize=(10, 6))

    # Create bars
    bar1 = plt.bar(index, totals_no_opt, bar_width, label="Without Optimization", color="#1f77b4")
    bar2 = plt.bar(index + bar_width, totals_with_opt, bar_width, label="With Optimization", color="#ff7f0e")

    # Add labels and title
    plt.xlabel("Configuration")
    plt.ylabel("Total Energy Consumption")
    plt.title("Energy Consumption Comparison (With vs Without Optimization)")
    plt.xticks(index + bar_width / 2, configurations)
    plt.legend()

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:,.0f}", ha="center", va="bottom")

    add_labels(bar1)
    add_labels(bar2)

    plt.tight_layout()
    plt.show()


def plot_energy(dataset="mnist"):
    if dataset == "mnist":
        data_W = load("MNIST_PERF_ROUNDS_251_BASE2.pkl")
        data_WP = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
        data_WP_G1 = load("MNIST_PERF_ROUNDS_251_G1.pkl")
        data = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
    else:
        data_W = load("CIFAR_PERF_ROUNDS_251_BASE.pkl")
        data_WP = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")
        data_WP_G1 = load("CIFAR_PERF_ROUNDS_251_G1.pkl")
        data = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")

    base_cmp = np.sum([vals[0] for vals in data["energy"].values()]) * 100 / 251
    fmax_cmp = np.sum([vals[0] for vals in data["energy"].values()]) * 90 / 251
    fopt_cmp = np.sum([vals[1] for vals in data["energy"].values()]) * 110 / 251  #

    base_trs = np.sum([vals[2] for vals in data_W["energy"].values()]) / 251 * 100
    wp_g1 = np.sum([vals[3] for vals in data_WP_G1["energy"].values()]) / 251 * 90
    wp_gopt = np.sum([vals[3] for vals in data_WP["energy"].values()]) / 251 * 110
    trans_energy = {  # mnist old
        "$W, \gamma=1, f_{2GHz}$": base_cmp + base_trs,
        "$W_p, \gamma_{opt}, f_{2GHz}$": fmax_cmp + wp_gopt,
        "$W_p, \gamma=1, f_{opt}$": wp_g1 + fopt_cmp,
        "$W_p, \gamma_{opt}, f_{opt}$": fopt_cmp + wp_gopt,
    }

    labels = list(trans_energy.keys())
    values = list(trans_energy.values())
    plt.figure()
    colors = ["skyblue", "skyblue", "skyblue", "lightgreen", "skyblue", "red"]
    # colors = ['skyblue'] * 7
    # vals = ['100\nRounds', '100\nRounds', '160\nRounds', '220\nRounds', '380\nRounds', '500\nRounds', '500 \nRounds']
    if dataset == "mnist":
        vals = ["100\nRounds", "70\nRounds", "140\nRounds", "180\nRounds", "360\nRounds", "450\nRounds", "500 \nRounds"]
    else:
        vals = [
            "170\nRounds",
            "180\nRounds",
            "500\nRounds",
            "180\nRounds",
            "360\nRounds",
            "450\nRounds",
            "500 \nRounds",
        ]
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    # for bar, value in zip(bars, vals):
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2,  # Center of the bar
    #         bar.get_height() / 2,  # Vertical position inside the bar
    #         f"{value}",  # Format the value to 4 decimal places
    #         ha='center',  # Horizontal alignment
    #         va='center',  # Vertical alignment
    #         color='black',  # Text color
    #         fontsize=10
    #     )

    # Adding title and labels
    plt.grid(linestyle="dashed")
    plt.xlabel("Network configurations", fontsize=14)
    plt.ylabel("Total Training Energy [J]", fontsize=13)
    # Customize x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=13)
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Show the plot
    plt.show()


def gamma(epsilon=0.2, lambda_param=0.005):
    # Parameters (adjust ε and λ as desired)

    # Domain
    x = np.linspace(0, 500, 400)

    # Lower-bounded exponential decay
    # y = epsilon + (1 - epsilon) * np.exp(-lambda_param * x)
    y = epsilon + (1 - epsilon) * np.exp(-lambda_param * x)

    # Plot
    plt.figure()
    plt.plot(x, y, label=f"ε={epsilon}\nλ={lambda_param}")
    plt.ylim(0, 1.05)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Lower-Bounded Exponential Decay: $f(x) = \epsilon + (1-\epsilon) \exp^{−\lambda x}$")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_energy_stacked(dataset="mnist"):
    if dataset == "mnist":
        data_W = load("MNIST_PERF_ROUNDS_251_BASE2.pkl")
        data_WP = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
        data_WP_G1 = load("MNIST_PERF_ROUNDS_251_G1.pkl")
        data = load("MNIST_PERF_ROUNDS_251_OPT.pkl")
    else:
        data_W = load("CIFAR_PERF_ROUNDS_251_BASE.pkl")
        data_WP = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")
        data_WP_G1 = load("CIFAR_PERF_ROUNDS_251_G1.pkl")
        data = load("CIFAR_PERF_ROUNDS_251_OPT.pkl")

    # Group 1 (compute energy)
    base_cmp = np.sum([vals[0] for vals in data["energy"].values()]) * 100 / 251
    fmax_cmp = np.sum([vals[0] for vals in data["energy"].values()]) * 90 / 251
    fopt_cmp = 5 + np.sum([vals[1] for vals in data["energy"].values()]) * 110 / 251

    # Group 2 (transmission energy)
    base_trs = np.sum([vals[2] for vals in data_W["energy"].values()]) / 251 * 100
    wp_g1 = np.sum([vals[3] for vals in data_WP_G1["energy"].values()]) / 251 * 90
    wp_gopt = np.sum([vals[3] for vals in data_WP["energy"].values()]) / 251 * 110
    print(f"wp_gopt={wp_gopt}")
    print(f"fopt_cmp={fopt_cmp}")
    configs = [
        (r"$\boldsymbol{W}, \gamma=1, \boldsymbol{f}_{2GHz}$", base_cmp, base_trs),
        (r"$\boldsymbol{W_p}, \gamma_{OPT}, \boldsymbol{f}_{2GHz}$", fmax_cmp, wp_gopt),
        (r"$\boldsymbol{W_p}, \gamma=1, \boldsymbol{f}_{OPT}$", fopt_cmp, wp_g1),
        (r"$\boldsymbol{W_p}, \gamma_{OPT}, \boldsymbol{f}_{OPT}$", fopt_cmp, wp_gopt),
    ]
    colors = ["skyblue", "skyblue", "skyblue", "lightgreen", "skyblue", "red"]
    labels = [cfg[0] for cfg in configs]
    compute_vals = [cfg[1] for cfg in configs]
    trans_vals = [cfg[2] for cfg in configs]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots()

    # Bar for compute energy (bottom)
    bar1 = ax.bar(x, compute_vals, width, label="Computation Energy", color=colors, hatch="", edgecolor="black")

    # Bar for transmission energy (on top)
    bar2 = ax.bar(
        x,
        trans_vals,
        width,
        bottom=compute_vals,
        label="Transmission Energy",
        color=colors,
        hatch="*",
        edgecolor="black",
    )

    print([compute_vals[i]+trans_vals[i] for i in range(len(trans_vals))])

    ax.set_xlabel("Network configurations", fontsize=14)
    ax.set_ylabel("Total Training Energy [J]", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle="dashed")
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", hatch="", label="Compute Energy"),
        Patch(facecolor="white", edgecolor="black", hatch="*", label="Transmission Energy"),
    ]
    ax.legend(handles=legend_elements, fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # w_heatmaps(1, 2)
    # ===>> 1-Rounds
    # plot_rounds(dataset="mnist")
    # plot_rounds(dataset="cifar")
    # ===>> 2-Gamma_Time
    # plot_time(dataset='mnist', target_acc=0.9)
    # plot_time(dataset='cifar', target_acc=0.68)
    # ===>> 3-Freq_Time
    # plot_freq_time(dataset="mnist", target_accuracy=0.91)
    # plot_freq_time(dataset="cifar", target_accuracy=0.7105, ylim=0.8)
    # ===>> 4-TransEnergy
    # plot_trans_energy(dataset="mnist")
    plot_trans_energy(dataset="cifar")
    # ===>> 5-ComputeEnergy
    # plot_compute_energy(dataset="mnist")
    # plot_compute_energy(dataset="cifar")
    # ===>> 6-Energy
    # plot_energy(dataset="mnist")
    # plot_energy_stacked(dataset="mnist")
    # plot_energy_stacked(dataset="cifar")
    # plot_energy(dataset="cifar")
    # plot_energy_comparison(dataset="mnist")
    # plot_accuracy_vs_time()
    # plot_total_energy()
    # gamma(epsilon=0.1, lambda_param=0.005)
    pass
