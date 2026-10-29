"""
CPU5006: Artificial Intelligence
Assessment S1: Rules-Based Artificial Intelligence

Car Evaluation Dataset
Rule-Based AI Classification System

This program:
1. Extracts the supplied Car Evaluation Dataset ZIP file.
2. Locates and loads car.data.
3. Assigns meaningful column names.
4. Performs basic data validation and analysis.
5. Splits the dataset into training and testing data.
6. Builds two rule-based classification algorithms.
7. Generates predictions.
8. Evaluates the algorithms using accuracy, precision, recall and F1-score.
9. Produces confusion matrices.
10. Demonstrates predictions using sample cars.

The rule-based methods are implemented using explicit IF-THEN logic.

"""

import os
import zipfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ZIP_FILE = "Car Evaluation Dataset(1) 2.zip"

EXTRACT_FOLDER = "car_evaluation_extracted"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. EXTRACT DATASET
# ============================================================

def extract_dataset(zip_file, extract_folder):
    """
    Extract the supplied ZIP dataset if it has not already
    been extracted.

    Returns the path to the car.data file.
    """

    if not os.path.exists(zip_file):
        raise FileNotFoundError(
            f"Dataset ZIP file was not found: {zip_file}\n"
            f"Place the ZIP file in the same folder as this Python program."
        )

    if not os.path.exists(extract_folder):
        print("Extracting dataset ZIP file...")
        os.makedirs(extract_folder)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        print("Dataset extracted successfully.")
    else:
        print("Dataset extraction folder already exists.")

    car_data_path = None

    for root, directories, files in os.walk(extract_folder):
        for file in files:
            if file == "car.data":
                car_data_path = os.path.join(root, file)
                break

        if car_data_path is not None:
            break

    if car_data_path is None:
        raise FileNotFoundError(
            "car.data could not be found inside the ZIP file."
        )

    return car_data_path


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset(car_data_path):
    """
    Load the original Car Evaluation car.data file.

    The dataset contains six input attributes and one target
    attribute.
    """

    columns = [
        "buying",
        "maintenance",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class"
    ]

    df = pd.read_csv(
        car_data_path,
        header=None,
        names=columns
    )

    return df


# ============================================================
# 4. CLEAN DATA
# ============================================================

def clean_dataset(df):
    """
    Perform basic cleaning and validation.
    """

    print("\nChecking dataset...")

    # Remove leading/trailing whitespace
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()

    # Remove completely duplicated records
    duplicate_count = df.duplicated().sum()

    print("Duplicate records:", duplicate_count)

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # Check missing values
    missing_values = df.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    return df


# ============================================================
# 5. DATASET OVERVIEW
# ============================================================

def display_dataset_overview(df):
    """
    Display useful information about the dataset.
    """

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nTarget class distribution:")
    print(df["class"].value_counts())

    print("\nUnique values:")
    for column in df.columns:
        print(f"\n{column}:")
        print(df[column].unique())


# ============================================================
# 6. EXPLORATORY ANALYSIS
# ============================================================

def perform_basic_eda(df):
    """
    Perform basic frequency analysis to understand
    the categorical dataset before designing rules.
    """

    print("\n" + "=" * 70)
    print("BASIC EXPLORATORY ANALYSIS")
    print("=" * 70)

    for column in df.columns:
        print("\n" + column.upper())
        print(df[column].value_counts())

    # Target distribution graph
    plt.figure(figsize=(8, 5))

    df["class"].value_counts().plot(
        kind="bar"
    )

    plt.title("Car Evaluation Class Distribution")
    plt.xlabel("Car Evaluation")
    plt.ylabel("Number of Cars")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "class_distribution.png",
        dpi=300
    )

    plt.show()

    # Safety versus class
    safety_table = pd.crosstab(
        df["safety"],
        df["class"]
    )

    print("\nSafety versus Car Evaluation:")
    print(safety_table)

    # Buying versus class
    buying_table = pd.crosstab(
        df["buying"],
        df["class"]
    )

    print("\nBuying Price versus Car Evaluation:")
    print(buying_table)

    # Persons versus class
    persons_table = pd.crosstab(
        df["persons"],
        df["class"]
    )

    print("\nNumber of Persons versus Car Evaluation:")
    print(persons_table)


# ============================================================
# 7. TRAIN/TEST SPLIT
# ============================================================

def split_dataset(df):
    """
    Split the dataset into training and testing subsets.

    The training data is used to analyse the data and develop
    the rule-based system.

    The testing data is kept separate for final evaluation.
    """

    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


# ============================================================
# 8. RULE-BASED ALGORITHM 1
# ============================================================

def rule_based_algorithm_1(row):
    """
    Rule-Based Algorithm 1.

    This is a priority-based expert-style rule system.

    The rules evaluate safety and passenger capacity first,
    followed by buying and maintenance conditions.

    Each rule is an explicit IF-THEN decision.
    """

    buying = row["buying"]
    maintenance = row["maintenance"]
    persons = row["persons"]
    lug_boot = row["lug_boot"]
    safety = row["safety"]

    # --------------------------------------------------------
    # Rule 1
    # Low safety is treated as unacceptable.
    # --------------------------------------------------------
    if safety == "low":
        return "unacc"

    # --------------------------------------------------------
    # Rule 2
    # A two-person capacity is treated as unacceptable.
    # --------------------------------------------------------
    if persons == "2":
        return "unacc"

    # --------------------------------------------------------
    # Rule 3
    # High safety, four persons and a large boot provide
    # favourable conditions when the buying price is low.
    # --------------------------------------------------------
    if (
        safety == "high"
        and persons == "4"
        and lug_boot == "big"
        and buying == "low"
        and maintenance in ["low", "med"]
    ):
        return "vgood"

    # --------------------------------------------------------
    # Rule 4
    # High safety and good passenger capacity with low
    # buying price gives a good evaluation.
    # --------------------------------------------------------
    if (
        safety == "high"
        and persons in ["4", "more"]
        and buying == "low"
    ):
        return "good"

    # --------------------------------------------------------
    # Rule 5
    # High safety and medium buying price with reasonable
    # maintenance can be acceptable.
    # --------------------------------------------------------
    if (
        safety == "high"
        and persons in ["4", "more"]
        and buying == "med"
        and maintenance in ["low", "med"]
    ):
        return "acc"

    # --------------------------------------------------------
    # Rule 6
    # Medium safety with good capacity and low buying price
    # can receive an acceptable evaluation.
    # --------------------------------------------------------
    if (
        safety == "med"
        and persons in ["4", "more"]
        and buying == "low"
        and maintenance == "low"
    ):
        return "acc"

    # --------------------------------------------------------
    # Rule 7
    # High maintenance combined with higher buying price
    # generally reduces the evaluation.
    # --------------------------------------------------------
    if (
        maintenance == "high"
        and buying == "high"
    ):
        return "unacc"

    # --------------------------------------------------------
    # Default rule
    # --------------------------------------------------------
    return "acc"


# ============================================================
# 9. RULE-BASED ALGORITHM 2
# ============================================================

def rule_based_algorithm_2(row):
    """
    Rule-Based Algorithm 2.

    This algorithm uses a scoring-style rule system.

    Positive conditions add points.
    Negative conditions reduce points.

    The final score is converted into one of the four
    Car Evaluation categories.
    """

    buying = row["buying"]
    maintenance = row["maintenance"]
    persons = row["persons"]
    lug_boot = row["lug_boot"]
    safety = row["safety"]

    score = 0

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if safety == "high":
        score += 3
    elif safety == "med":
        score += 1
    elif safety == "low":
        score -= 4

    # --------------------------------------------------------
    # Number of persons
    # --------------------------------------------------------

    if persons == "more":
        score += 3
    elif persons == "4":
        score += 2
    elif persons == "2":
        score -= 3

    # --------------------------------------------------------
    # Buying price
    # --------------------------------------------------------

    if buying == "low":
        score += 2
    elif buying == "med":
        score += 1
    elif buying == "high":
        score -= 2

    # --------------------------------------------------------
    # Maintenance price
    # --------------------------------------------------------

    if maintenance == "low":
        score += 2
    elif maintenance == "med":
        score += 1
    elif maintenance == "high":
        score -= 2

    # --------------------------------------------------------
    # Boot size
    # --------------------------------------------------------

    if lug_boot == "big":
        score += 2
    elif lug_boot == "med":
        score += 1
    elif lug_boot == "small":
        score -= 1

    # --------------------------------------------------------
    # Convert score to final class
    # --------------------------------------------------------

    if score <= 0:
        return "unacc"

    elif score <= 3:
        return "acc"

    elif score <= 6:
        return "good"

    else:
        return "vgood"


# ============================================================
# 10. GENERATE PREDICTIONS
# ============================================================

def generate_predictions(X_test):
    """
    Generate predictions using both rule-based algorithms.
    """

    predictions_algorithm_1 = X_test.apply(
        rule_based_algorithm_1,
        axis=1
    )

    predictions_algorithm_2 = X_test.apply(
        rule_based_algorithm_2,
        axis=1
    )

    return (
        predictions_algorithm_1,
        predictions_algorithm_2
    )


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_algorithm(y_true, y_pred, algorithm_name):
    """
    Calculate classification performance metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(algorithm_name)
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return {
        "Algorithm": algorithm_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_true, y_pred, algorithm_name):
    """
    Create and save a confusion matrix.
    """

    labels = ["unacc", "acc", "good", "vgood"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"Confusion Matrix - {algorithm_name}"
    )

    plt.colorbar()

    tick_marks = np.arange(len(labels))

    plt.xticks(
        tick_marks,
        labels
    )

    plt.yticks(
        tick_marks,
        labels
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    # Add values inside the matrix
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    filename = (
        algorithm_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()


# ============================================================
# 13. COMPARE ALGORITHMS
# ============================================================

def compare_algorithms(results):
    """
    Display and visualise the performance comparison.
    """

    comparison = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    # Save results
    comparison.to_csv(
        "rule_based_performance_results.csv",
        index=False
    )

    # Plot comparison
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    plot_data = comparison.set_index(
        "Algorithm"
    )[metrics]

    ax = plot_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Comparison of Rule-Based AI Algorithms"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "algorithm_comparison.png",
        dpi=300
    )

    plt.show()

    return comparison


# ============================================================
# 14. TEST INDIVIDUAL CAR
# ============================================================

def test_individual_car(
    buying,
    maintenance,
    doors,
    persons,
    lug_boot,
    safety
):
    """
    Test a single car using both rule-based algorithms.
    """

    car = pd.Series({
        "buying": buying,
        "maintenance": maintenance,
        "doors": doors,
        "persons": persons,
        "lug_boot": lug_boot,
        "safety": safety
    })

    prediction_1 = rule_based_algorithm_1(car)

    prediction_2 = rule_based_algorithm_2(car)

    print("\n" + "=" * 70)
    print("INDIVIDUAL CAR TEST")
    print("=" * 70)

    print("Buying price :", buying)
    print("Maintenance  :", maintenance)
    print("Doors        :", doors)
    print("Persons      :", persons)
    print("Boot         :", lug_boot)
    print("Safety       :", safety)

    print("\nAlgorithm 1 prediction:", prediction_1)
    print("Algorithm 2 prediction:", prediction_2)


# ============================================================
# 15. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("CPU5006 - RULE-BASED AI")
    print("CAR EVALUATION DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Extract ZIP
    # --------------------------------------------------------

    car_data_path = extract_dataset(
        ZIP_FILE,
        EXTRACT_FOLDER
    )

    print("\nDataset file located at:")
    print(car_data_path)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_dataset(
        car_data_path
    )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = clean_dataset(
        df
    )

    # --------------------------------------------------------
    # Display overview
    # --------------------------------------------------------

    display_dataset_overview(
        df
    )

    # --------------------------------------------------------
    # Basic EDA
    # --------------------------------------------------------

    perform_basic_eda(
        df
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        df
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    (
        predictions_1,
        predictions_2
    ) = generate_predictions(
        X_test
    )

    # --------------------------------------------------------
    # Evaluate Algorithm 1
    # --------------------------------------------------------

    results_1 = evaluate_algorithm(
        y_test,
        predictions_1,
        "Rule-Based Algorithm 1"
    )

    # --------------------------------------------------------
    # Algorithm 1 confusion matrix
    # --------------------------------------------------------

    plot_confusion_matrix(
        y_test,
        predictions_1,
        "Rule-Based Algorithm 1"
    )

    # --------------------------------------------------------
    # Evaluate Algorithm 2
    # --------------------------------------------------------

    results_2 = evaluate_algorithm(
        y_test,
        predictions_2,
        "Rule-Based Algorithm 2"
    )

    # --------------------------------------------------------
    # Algorithm 2 confusion matrix
    # --------------------------------------------------------

    plot_confusion_matrix(
        y_test,
        predictions_2,
        "Rule-Based Algorithm 2"
    )

    # --------------------------------------------------------
    # Compare algorithms
    # --------------------------------------------------------

    comparison = compare_algorithms(
        [
            results_1,
            results_2
        ]
    )

    # --------------------------------------------------------
    # Display sample predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    sample_results = X_test.copy()

    sample_results["Actual"] = y_test.values

    sample_results["Algorithm_1"] = predictions_1.values

    sample_results["Algorithm_2"] = predictions_2.values

    print(
        sample_results.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Test an individual car
    # --------------------------------------------------------

    test_individual_car(
        buying="low",
        maintenance="low",
        doors="4",
        persons="4",
        lug_boot="big",
        safety="high"
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROGRAM COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print("- class_distribution.png")
    print("- rule_based_algorithm_1_confusion_matrix.png")
    print("- rule_based_algorithm_2_confusion_matrix.png")
    print("- algorithm_comparison.png")
    print("- rule_based_performance_results.csv")

    print("\nThe rule-based AI evaluation has completed successfully.")


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()