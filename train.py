#!/usr/bin/env python3
# coding: utf-8

import math
import sys
import time
import csv
import argparse
from constants import (
    DEFAULT_DATA_FILE,
    DEFAULT_THETA_FILE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_ITER,
    MIN_ROWS_FOR_TRAINING,
    MIN_DISTINCT_MILEAGES,
    LOW_DATA_WARNING_THRESHOLD,
    ANSI_RESET,
    ANSI_WARNING,
    CONVERGENCE_THRESHOLD
)

def read_dataset(filename):
    """
    Reads the dataset from a CSV file and returns two lists:
    mileages and prices.
    """
    mileages = []
    prices = []
    try:
        with open(filename, "r", newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        if len(rows) == 0:
            print(
                f"Error: Invalid CSV format. File is empty. "
                f"At least a header and {MIN_ROWS_FOR_TRAINING} data rows are required."
            )
            sys.exit(1)

        csv_header = [col.strip().lower() for col in rows[0]]
        if len(csv_header) != 2:
            print("Error: Invalid CSV format. Header row must contain 2 columns (example: km,price).")
            sys.exit(1)

        if csv_header[0] != "km" or csv_header[1] != "price":
            print("Error: Invalid CSV format. Header row must be 'km,price'.")
            sys.exit(1)

        for line_no, row in enumerate(rows[1:], start=2):
            if len(row) ==0 or all(cell.strip() == "" for cell in row):
                continue

            if len(row) != 2:
                print(f"Error: Invalid CSV format. Line {line_no}: expected 2 columns, found {len(row)}.")
                sys.exit(1)

            km_text = row[0].strip()
            price_text = row[1].strip()

            if km_text == "" or price_text == "":
                if km_text == "":
                    print(f"Error: Invalid CSV format. Line {line_no}: km_text cannot be empty.")
                if price_text == "":
                    print(f"Error: Invalid CSV format. Line {line_no}: price_text cannot be empty.")
                sys.exit(1)

            mileage = 0.0
            price = 0.0
            try:
                mileage = float(km_text)
                price = float(price_text)
            except ValueError:
                print(f"Error: Invalid CSV format. Line {line_no}: km and price must be numeric.")
                sys.exit(1)
                
            if not math.isfinite(mileage) or not math.isfinite(price):
                print(
                    f"Error: Invalid CSV format. Line {line_no}: "
                    "km and price must be finite numbers (not NaN/Inf)."
                )
                sys.exit(1)
            
            if mileage < 0:
                print(f"Error: Invalid CSV format. Line {line_no}: km cannot be negative.")
                sys.exit(1)
            
            if price < 0:
                print(f"Error: Invalid CSV format. Line {line_no}: price cannot be negative.")
                sys.exit(1)

            mileages.append(mileage)
            prices.append(price)          

        if len(mileages) < MIN_ROWS_FOR_TRAINING:
            print(
                f"Error: Invalid CSV format. At least {MIN_ROWS_FOR_TRAINING} "
                "data rows are required for linear regression.")
            sys.exit(1)

        unique_mileages = set(mileages)
        if len(unique_mileages) < MIN_DISTINCT_MILEAGES:
            print(
                f"Error: Invalid CSV format. At least {MIN_DISTINCT_MILEAGES} "
                "distinct km values are required (the slope cannot be learned from identical km values).")
            sys.exit(1)

        return mileages, prices

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Could not open {filename} (permission denied).")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error: Invalid CSV format. CSV parse error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Data reading error: {e}")
        sys.exit(1)

def parse_arguments():
    """
    Parses command-line arguments.
    
    Supported arguments:
        --learning-rate: Learning rate (default: 0.1)
        --max-iter: Maximum number of iterations (default: 10000)
        --help: Help message
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        usage="\"python train.py [OPTIONS]\" or \"./train.py [OPTIONS]\"",
        description="Train linear regression model with gradient descent.",
        epilog=(
            "Examples:\n"
            "  python train.py\n"
            "  python train.py --learning-rate 0.01\n"
            "  python train.py --learning-rate 0.05 --max-iter 5000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Set learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Set maximum iterations (default: {DEFAULT_MAX_ITER})",
    )
    
    args = parser.parse_args()

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")

    if args.max_iter <= 0:
        parser.error("--max-iter must be positive.")

    return args.learning_rate, args.max_iter

def normalize_data(data):
    """
    Normalizes the given data using Z-score normalization.

    Z-score normazlization formula: (x - mean) / std_dev

    Returns:
        normalized_data: Normalized data
        mean: Mean value of given data
        std_dev: Standard deviation of given data
    """

    total = 0
    for value in data:
        total += value
    mean = total / len(data)

    squarred_diff_sum = 0
    for value in data:
        diff = value - mean
        squarred_diff_sum += diff * diff
    variance = squarred_diff_sum / len(data)
    std_dev = variance ** 0.5

    if std_dev == 0:  # means data is a constant array
        normalized_data = [0.0] * len(data)
    else:
        normalized_data = []
        for value in data:
            normalized_value = (value - mean) / std_dev
            normalized_data.append(normalized_value)
    
    return normalized_data, mean, std_dev

def calculate_cost(mileages, prices, theta0, theta1):
    """
    Calculates the cost (mean squared error) of an approximation given by the parameters.

    Cost function formula: J(theta0, theta1) = (1/2m) * sum((prediction - actual)^2)

    Returns:
        cost: The calculated cost value
        Smaller cost values indicate a better fit of the model to the data.
    """
    m = len(mileages)
    total_error = 0.0

    for i in range(m):
        prediction = theta0 + theta1 * mileages[i]
        error = prediction - prices[i]
        total_error += error * error

    cost = total_error / (2 * m)
    return cost     

def train_model(mileages, prices, learning_rate, max_iter):
    """
        Trains a linear regression model using gradient descent
        according to the specified dataset.
        
        Parameters:
            mileages: List of mileage values (normalized)
            prices: List of price values (normalized)
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations for training
    """
    m = len(mileages)
    theta0 = 0.0
    theta1 = 0.0

    previous_cost = float('inf')
    no_improvement_count = 0
    patience = 50

    def is_check_iteration(iteration):
        if iteration <= 500:
            return iteration % 50 == 0
        else:
            return iteration % 10 == 0

    print("\nModel training started...")
    print(f"Data points: {m}")
    print(f"Learning rate: {learning_rate}")
    print(f"Maximum iterations: {max_iter}")
    print("-" * 50)  

    for iteration in range(1, max_iter + 1):
        sum_error_theta0 = 0.0
        sum_error_theta1 = 0.0

        for i in range(m):
            prediction = theta0 + theta1 * mileages[i]
            error = prediction - prices[i]

            sum_error_theta0 += error
            sum_error_theta1 += error * mileages[i]

        temp_theta0 = learning_rate * (sum_error_theta0 / m)
        temp_theta1 = learning_rate * (sum_error_theta1 / m)
            
        theta0 -= temp_theta0
        theta1 -= temp_theta1

        if is_check_iteration(iteration) or iteration == max_iter:
            current_cost = calculate_cost(mileages, prices, theta0, theta1)
            if iteration % 500 == 0:
                print(f"Iteration {iteration:5d}: Cost = {current_cost:.6f}")

            # Check for convergence
            cost_diff = abs(previous_cost - current_cost)
            if cost_diff < CONVERGENCE_THRESHOLD:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"\nEarly stopping ! Convergence reached at iteration {iteration}.")
                    print(f"Cost change {cost_diff:.6f} is less than threshold {CONVERGENCE_THRESHOLD}.")
                    break
            else:
                no_improvement_count = 0
            
            previous_cost = current_cost
    
    print("-" * 50)
    print("Training completed!")
    final_cost = calculate_cost(mileages, prices, theta0, theta1)
    print(f"Final cost: {final_cost:.6f}")
    
    return theta0, theta1

def denormalize_thetas(theta0_normed, theta1_normed, mile_mean, mile_std, price_mean, price_std):
    """
    Converts normalized theta values back to the original scale.
    
    Normalization:
        x_norm = (x - mean_x) / std_x
        y_norm = (y - mean_y) / std_y
    
    Model (normalized):
        y_norm = θ0_norm + θ1_norm * x_norm
    
    Denormalized:
        y = θ0_real + θ1_real * x
    
    Conversion formulas:
        θ1_real = θ1_norm * (std_y / std_x)
        θ0_real = mean_y + θ0_norm * std_y - θ1_real * mean_x
    """

    if mile_std != 0:
        theta1_denormed = theta1_normed * (price_std / mile_std)
        theta0_denormed = price_mean - theta1_denormed * mile_mean + (theta0_normed * price_std)
    else:
        theta1_denormed = 0.0
        theta0_denormed = price_mean

    return theta0_denormed, theta1_denormed

def save_thetas(theta0, theta1):
    """
    Saves θ0 and θ1 values to a CSV file.
    """
    try:
        with open(DEFAULT_THETA_FILE, 'w') as file:
            file.write("theta0,theta1\n")
            file.write(f"{theta0},{theta1}\n")
        print(f"\nθ values saved to '{DEFAULT_THETA_FILE}'.")
        print(f"θ0 = {theta0}")
        print(f"θ1 = {theta1}")
    except Exception as e:
        print(f"Error: Failed to save theta values: {e}")
        sys.exit(1)

def main():
    """
    Main training program function
    """
    learning_rate, max_iter = parse_arguments()

    print("\nReading data set...")
    mileages, prices = read_dataset(DEFAULT_DATA_FILE)
    print(f"Loaded {len(mileages)} data points.")

    if len(mileages) < LOW_DATA_WARNING_THRESHOLD:
        print(
            f"{ANSI_WARNING}Warning: Your dataset is small "
            f"(There should be at least {LOW_DATA_WARNING_THRESHOLD} rows).{ANSI_RESET}"
        )
        print(f"{ANSI_WARNING}Technical note: The model will run, but more data is recommended for robust and generalizable training.{ANSI_RESET}\n")
        time.sleep(3)

    print("\n Normalizing data...")
    mileages_normed, mileage_mean, mileage_std_dev = normalize_data(mileages)
    prices_normed, price_mean, price_std_dev = normalize_data(prices)

    print("\nTraining model...")
    theta0_normed, theta1_normed = train_model(mileages_normed, prices_normed, learning_rate, max_iter)


    print("\nDenormalizing theta values...")
    theta0, theta1 = denormalize_thetas(theta0_normed, theta1_normed, mileage_mean, mileage_std_dev, price_mean, price_std_dev)

    save_thetas(theta0, theta1)

        
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("To make predictions: python predict.py")
    print("=" * 50)


if __name__ == "__main__":
    main()