#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_no_normalization.py - Experimental raw-scale training program
Trains the model using linear regression and gradient descent without applying
feature/target normalization first. This file exists to demonstrate why
normalization matters for the original project.

Usage:
    python train_no_normalization.py                   # With default settings
    python train_no_normalization.py --learning-rate 0.01
    python train_no_normalization.py --max-iter 5000
    python train_no_normalization.py --learning-rate 1e-12 --max-iter 500000
"""

import sys
import csv
import time
import argparse
from constants import (
    DEFAULT_DATA_FILE,
    MIN_ROWS_FOR_TRAINING,
    MIN_DISTINCT_MILEAGES,
    LOW_DATA_WARNING_THRESHOLD,
    ANSI_RESET,
    ANSI_WARNING,
)

RAW_SCALE_THETA_FILE = "thetas_no_normalization.csv"
RAW_SCALE_DEFAULT_LEARNING_RATE = 1e-11
RAW_SCALE_PROGRESS_EVERY = 10000
RAW_SCALE_RELATIVE_CONVERGENCE_THRESHOLD = 1e-9
RAW_SCALE_PATIENCE = 20


def read_data(filename):
    """
    Reads data from a CSV file.
    
    Parameters:
        filename: Name of the CSV file
    
    Returns:
        mileages: List of mileages
        prices: List of prices
    """
    mileages = []
    prices = []

    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        if len(rows) == 0:
            print(
                f"Error: Invalid CSV format. File is empty. "
                f"At least a header and {MIN_ROWS_FOR_TRAINING} data rows are required."
            )
            sys.exit(1)

        # Validate the header row
        header = [col.strip().lower() for col in rows[0]]
        if len(header) != 2:
            print("Error: Invalid CSV format. Header row must contain 2 columns (example: km,price).")
            sys.exit(1)

        if header[0] != "km" or header[1] != "price":
            print("Error: Invalid CSV format. Header row must be 'km,price'.")
            sys.exit(1)

        # Read and validate data rows
        for line_no, row in enumerate(rows[1:], start=2):
            # Skip completely empty rows
            if len(row) == 0 or all(cell.strip() == "" for cell in row):
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

            if mileage < 0:
                print(f"Error: Invalid CSV format. Line {line_no}: km cannot be negative.")
                sys.exit(1)
            
            if price < 0:
                print(f"Error: Invalid CSV format. Line {line_no}: price cannot be negative.")
                sys.exit(1)

            mileages.append(mileage)
            prices.append(price)

        # Minimum data requirement for linear regression
        if len(mileages) < MIN_ROWS_FOR_TRAINING:
            print(
                f"Error: Invalid CSV format. At least {MIN_ROWS_FOR_TRAINING} "
                "data rows are required for linear regression."
            )
            sys.exit(1)

        unique_mileages = set(mileages)
        if len(unique_mileages) < MIN_DISTINCT_MILEAGES:
            print(
                f"Error: Invalid CSV format. At least {MIN_DISTINCT_MILEAGES} distinct km values "
                "are required (the slope cannot be learned from identical km values)."
            )
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


def normalize_data(data):

    # Compute the mean
    total = 0
    for value in data:
        total += value
    mean = total / len(data)
    
    # Compute the standard deviation
    # std = sqrt(sum((x - mean)^2) / n)
    sum_squared_diff = 0
    for value in data:
        diff = value - mean
        sum_squared_diff += diff * diff
    
    variance = sum_squared_diff / len(data)
    std_dev = variance ** 0.5  # Square root
    
    return data, mean, std_dev


def estimate_price(mileage, theta0, theta1):
    """
    Estimates the price.
    
    Formula: estimatePrice(mileage) = θ0 + (θ1 * mileage)
    """
    return theta0 + (theta1 * mileage)


def calculate_cost(mileages, prices, theta0, theta1):
    """
    Computes the cost function.
    
    Formula: J(θ) = (1/2m) * sum((h(x) - y)^2)
    
    This function indicates how well the model is performing.
    Smaller value = better model
    """
    m = len(mileages)
    total_error = 0
    
    for i in range(m):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = prediction - prices[i]
        total_error += error * error
    
    cost = total_error / (2 * m)
    return cost


def train_model(
    mileages,
    prices,
    learning_rate=RAW_SCALE_DEFAULT_LEARNING_RATE,
    max_iterations=None,
):
    """
    Trains the model using the gradient descent algorithm.
    
    Parameters:
        mileages: Raw mileage values
        prices: Raw price values
        learning_rate: Learning rate for raw-scale training
        max_iterations: Optional safety limit for raw-scale training
    
    Returns:
        theta0: Trained θ0 parameter
        theta1: Trained θ1 parameter
        completed: Whether training finished by convergence
    """
    m = len(mileages)  # Number of data points
    theta0 = 0.0
    theta1 = 0.0
    iteration = 0
    
    # Used for convergence checking
    previous_cost = float('inf')
    convergence_threshold = RAW_SCALE_RELATIVE_CONVERGENCE_THRESHOLD
    no_improvement_count = 0
    patience = RAW_SCALE_PATIENCE
    
    print("\nModel training started...")
    print(f"Data points: {m}")
    print(f"Learning rate: {learning_rate}")
    if max_iterations is None:
        print("Maximum iterations: unlimited (run until convergence or Ctrl+C)")
    else:
        print(f"Maximum iterations: {max_iterations}")
    print("-" * 50)
    
    try:
        # Gradient descent iterations
        while max_iterations is None or iteration < max_iterations:
            # Compute gradients
            sum_error_theta0 = 0
            sum_error_theta1 = 0
            
            # For every data point
            for i in range(m):
                # Make a prediction
                prediction = estimate_price(mileages[i], theta0, theta1)
                
                # Compute the error
                error = prediction - prices[i]
                
                # Accumulate gradient terms
                sum_error_theta0 += error
                sum_error_theta1 += error * mileages[i]
            
            # Compute temporary theta values
            temp_theta0 = learning_rate * (sum_error_theta0 / m)
            temp_theta1 = learning_rate * (sum_error_theta1 / m)
            
            # Update theta values simultaneously
            theta0 = theta0 - temp_theta0
            theta1 = theta1 - temp_theta1
            
            # Show progress periodically to keep raw-scale runs readable
            should_report = (
                iteration % RAW_SCALE_PROGRESS_EVERY == 0
                or (max_iterations is not None and iteration == max_iterations - 1)
            )
            if should_report:
                current_cost = calculate_cost(mileages, prices, theta0, theta1)
                print(f"Iteration {iteration:5d}: Cost = {current_cost:.6f}")
                
                # On the raw scale, compare relative improvement instead of
                # an absolute cost delta. Absolute costs can be huge.
                if previous_cost != float('inf'):
                    relative_cost_change = abs(previous_cost - current_cost) / max(abs(previous_cost), 1.0)
                    print(
                        " " * 17
                        + f"Relative cost change = {relative_cost_change:.12f}"
                    )
                    if relative_cost_change < convergence_threshold:
                        no_improvement_count += 1
                        if no_improvement_count >= patience:
                            print(f"\nEarly stopping! Iteration {iteration} - convergence reached.")
                            print(
                                f"Relative cost change: {relative_cost_change:.12f} "
                                f"< {convergence_threshold}"
                            )
                            print("-" * 50)
                            print("Training completed!")
                            print(f"Final cost: {current_cost:.6f}")
                            return theta0, theta1, True
                    else:
                        no_improvement_count = 0
                
                previous_cost = current_cost

            iteration += 1
    except KeyboardInterrupt:
        interrupted_cost = calculate_cost(mileages, prices, theta0, theta1)
        print("\n" + "-" * 50)
        print("Training interrupted by user.")
        print(f"Last completed iteration: {iteration}")
        print(f"Current cost: {interrupted_cost:.6f}")
        print("Convergence was not reached before interruption.")
        return theta0, theta1, False
            
    print("-" * 50)
    print("Training completed!")
    final_cost = calculate_cost(mileages, prices, theta0, theta1)
    print(f"Final cost: {final_cost:.6f}")
    
    return theta0, theta1, True


def denormalize_theta(theta0_norm, theta1_norm, mile_mean, mile_std, price_mean, price_std):

    return theta0_norm, theta1_norm


def save_thetas(theta0, theta1):
    """
    Saves θ0 and θ1 values to a separate CSV file so the original project
    outputs are not overwritten.
    """
    try:
        with open(RAW_SCALE_THETA_FILE, 'w') as file:
            file.write("theta0,theta1\n")
            file.write(f"{theta0},{theta1}\n")
        print(f"\nθ values saved to '{RAW_SCALE_THETA_FILE}'.")
        print(f"θ0 = {theta0}")
        print(f"θ1 = {theta1}")
    except Exception as e:
        print(f"Error: Failed to save theta values: {e}")
        sys.exit(1)


def parse_arguments_argparse():
    """
    Alternative command-line parser using argparse.
    This function is provided for comparison with the manual parser.

    Returns:
        learning_rate: Parsed learning rate
        max_iter: Parsed maximum iterations or None
    """
    parser = argparse.ArgumentParser(
        prog="train_no_normalization.py",
        usage="\"python train_no_normalization.py [OPTIONS]\" or \"./train_no_normalization.py [OPTIONS]\"",
        description="Train linear regression model with gradient descent on raw values.",
        epilog=(
            "Examples:\n"
            "  python train_no_normalization.py\n"
            "  python train_no_normalization.py --learning-rate 1e-11\n"
            "  python train_no_normalization.py --learning-rate 1e-12 --max-iter 500000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=RAW_SCALE_DEFAULT_LEARNING_RATE,
        help=f"Set learning rate (default: {RAW_SCALE_DEFAULT_LEARNING_RATE})",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Optional safety limit for iterations",
    )

    args = parser.parse_args()

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")

    if args.max_iter is not None and args.max_iter <= 0:
        parser.error("--max-iter must be positive.")

    return args.learning_rate, args.max_iter


def main():
    """
    Main program function.
    """
    print("=" * 50)
    print("RAW-SCALE MODEL TRAINING PROGRAM")
    print("=" * 50)
    
    # Parse arguments
    learning_rate, max_iter = parse_arguments_argparse()
    
    # Read the data
    print("\nReading data...")
    mileages, prices = read_data(DEFAULT_DATA_FILE)
    print(f"Loaded {len(mileages)} data points.")

    warning_color = ANSI_WARNING
    reset = ANSI_RESET

    if len(mileages) <= LOW_DATA_WARNING_THRESHOLD:
        print(
            f"{warning_color}Warning: Your dataset is small "
            f"(less than or equal to {LOW_DATA_WARNING_THRESHOLD} rows).{reset}"
        )
        print(f"{warning_color}Technical note: The model will run, but more data is recommended for robust and generalizable training.{reset}\n")
        time.sleep(3)
    
    # Intentionally skip normalization
    print("\nNormalization disabled: training on raw data values...")
    mileages_norm, mile_mean, mile_std = normalize_data(mileages)
    prices_norm, price_mean, price_std = normalize_data(prices)
    
    print(f"Mileage - Mean: {mile_mean:.2f}, Std: {mile_std:.2f}")
    print(f"Price - Mean: {price_mean:.2f}, Std: {price_std:.2f}")
    
    # Train the model directly on raw-scale data
    theta0_norm, theta1_norm, completed = train_model(
        mileages_norm, 
        prices_norm, 
        learning_rate=learning_rate,
        max_iterations=max_iter
    )

    if not completed:
        print("\nNo parameter file was written because training was stopped manually.")
        return
    
    # Identity step kept so the structure stays comparable to train.py
    print("\nSkipping denormalization step (parameters are already on the raw scale)...")
    theta0, theta1 = denormalize_theta(
        theta0_norm, theta1_norm,
        mile_mean, mile_std,
        price_mean, price_std
    )
    
    # Save the results
    save_thetas(theta0, theta1)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Raw-scale parameters were saved to: {RAW_SCALE_THETA_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
