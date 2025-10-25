import numpy as np
from . import constants

from google.cloud import storage
from pathlib import Path
import matplotlib.pyplot as plt
import os

def is_ascii(input_str):
    """
    A very basic function that checks if all elements of str are ASCII or not
    Args:
        input_str: input string
    """
    return all(ord(char) < 128 for char in input_str)


def diff_strings(str1, str2):
    """
    A function that returns the number of elements of two strings that are not identical
    Args:
        str1: the first string
        str2: the second string
    """
    if len(str1) != len(str2):
        print("Warning: length of two strings are not equal")
        return -1
    return sum(str1[i] != str2[i] for i in range(len(str1)))


def sigmoid(inp):
    """
    Computes the sigmoid function of a scalar or a 1d numpy array
    Args:
        inp: the input which can be a scalar or a 1d numpy array
    """
    inp = np.asarray(inp)
    # Checking for case when the input is an array/np.array of arrays. In this case only the first element of inp is
    # used. A common example is when A = np.array([np.array([1, 2, 3])]).
    if inp.ndim == 2:
        inp = inp[0]
    return 1.0 / (1.0 + np.exp(-np.clip(inp, -709.78, 709.78)))


def print_grapheme_clusters(thrsh, language, exclusive):
    """
    This function print the grapheme clusters and their frequencies for a given langauge. It also computes what
    percentage of grapheme clusters form which percent of the text
    Args:
        thrsh: shows what percent of the text we want to be covered by grapheme clusters
        language: shows the language that we are working with
        exclusive: shows if we only consider grapheme clusters in a single script or not
    """
    ratios = None
    if language == "Thai" and exclusive is False:
        ratios = constants.THAI_GRAPH_CLUST_RATIO
    if language == "Thai" and exclusive is True:
        ratios = constants.THAI_EXCLUSIVE_GRAPH_CLUST_RATIO
    if language == "Burmese" and exclusive is False:
        ratios = constants.BURMESE_GRAPH_CLUST_RATIO
    if language == "Burmese" and exclusive is True:
        ratios = constants.BURMESE_EXCLUSIVE_GRAPH_CLUST_RATIO
    if language == "Thai-Burmese":
        ratios = constants.THAI_BURMESE_GRAPH_CLUST_RATIO
    if ratios is None:
        print("No grapheme cluster dictionary has been computed for the input language.")
        return
    cum_sum = 0
    cnt = 0
    for val in ratios.values():
        cum_sum += val
        cnt += 1
        if cum_sum > thrsh:
            break
    print(ratios)
    print("number of different grapheme clusters in {} = {}".format(language, len(ratios.keys())))
    print("{} grapheme clusters form {} of the text".format(cnt, thrsh))


def download_from_gcs(gcs_uri: str, dir: str) -> None:
    """
    This function downloads data from Google Cloud Storage. Given gcs_uri, the function downloads 
    all subfolders and files inside and stores them in dir.
    Args:
        gcs_uri: A `gs://` URI that **must** contain both a bucket name and an object prefix, e.g. ``gs://my-bucket/Data/``.
        dir: Local root directory. Sub-folders identical to the GCS object paths are created beneath this location.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs://uri, got {gcs_uri}")
    gcs_uri = gcs_uri.rstrip('/')
    bucket_name, prefix = gcs_uri.replace('gs://', '').split('/', 1)
    prefix += '/' if not prefix.endswith('/') else ''

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for blob in bucket.list_blobs(prefix=prefix):
        rel_path = blob.name[len(prefix):]
        if rel_path == "":
            continue
        local_path = os.path.join(dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

def save_training_plot(history, model_path: Path) -> None:
    """
    Plot accuracy & loss curves from a Keras History object and save the figure.
    Args:
        history: The History returned by `model.fit(...)`.
        model_path: Destination directory, e.g. Path(__file__).parent.parent.absolute().joinpath("Models", model_name)
    """
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    ax_acc.plot(history.history["accuracy"],     label="train acc")
    ax_acc.plot(history.history["val_accuracy"], label="val acc")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    ax_loss.plot(history.history["loss"],     label="train loss", linestyle="--")
    ax_loss.plot(history.history["val_loss"], label="val loss",   linestyle="--")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    fig.suptitle("Training vs Validation — Accuracy & Loss")
    fig.tight_layout()

    model_path.mkdir(parents=True, exist_ok=True)
    out_file = model_path / "plot.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def upload_to_gcs(local_dir: str, gcs_uri: str) -> None:
    client = storage.Client()
    gcs_uri = gcs_uri.rstrip('/')
    bucket_name, *prefix_parts = gcs_uri.replace('gs://', '').split('/', 1)
    prefix = prefix_parts[0] if prefix_parts else ""
    bucket = client.bucket(bucket_name)
    for file_path in Path(local_dir).rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(Path(local_dir).parent).as_posix()
            blob_name = f"{prefix}/{rel_path}" if prefix else rel_path
            bucket.blob(blob_name).upload_from_filename(str(file_path))