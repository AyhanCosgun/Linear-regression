#!/usr/bin/env python3
# coding: utf-8

"""
precision.py - Model accuracy evaluation program (BONUS)
Evaluates the trained model using different metrics.

Usage:
    python precision.py                # Default: R² metric
    python precision.py --metric mse   # MSE metric
    python precision.py --metric rmse  # RMSE metric
    python precision.py --metric r2    # R² metric
"""

import argparse
from constants import DEFAULT_DATA_FILE
from train import read_dataset
from predict import read_trained_data

def calculate_mse(mileages, prices, theta_0, theta_1):
    """
    Computes MSE (Mean Squared Error).
    
    Formula: MSE = (1/m) * Σ(y_actual - y_predicted)²
    
    Meaning: Average squared error
    Unit: Square of the target variable (e.g. $²)
    Range: [0, ∞) , 0 is best
    """
    m = len(mileages)
    total_squared_error = 0.0

    for i in range(m):
        y_actual = prices[i]
        y_predicted = theta_0 + theta_1 * mileages[i]
        error = y_actual - y_predicted
        total_squared_error += error * error
    mse = total_squared_error / m
    return mse

def calculate_rmse(mileages, prices, theta_0, theta_1):
    """
    Computes RMSE (Root Mean Squared Error).
    
    Formula: RMSE = sqrt(MSE)
    
    Unit: Same as target variable (e.g. $)
    Range: [0, ∞) , 0 is best
    Interpretation: Average error magnitude
    """
    mse = calculate_mse(mileages, prices, theta_0, theta_1)
    rmse = mse ** 0.5
    return rmse

def calculate_r2(mileages, prices, theta_0, theta_1):
    """
    Computes R² (Coefficient of Determination).
    
    Formula: R² = 1 - (SS_res / SS_tot)
    
    Where:
    - SS_res = Σ(y_actual - y_predicted)²   # Residual sum of squares
    - SS_tot = Σ(y_actual - y_mean)²        # Total sum of squares (variance of the data)
    
    Unit: None (dimensionless)
    Range: (-∞, 1], 1 is best
    Interpretation: Proportion of variance of errors explained by the model
        1.0 = Perfect fit (100% explained)
        0.7 = Good fit (70% explained)
        0.0 = Model is as bad as predicting the mean
        <0  = Model is worse than predicting the mean
    """
    m = len(mileages)
    y_mean = sum(prices) / m

    ss_res = 0.0
    ss_tot = 0.0

    for i in range(m):
        y_actual = prices[i]
        y_predicted = theta_0 + theta_1 * mileages[i]
        ss_res += (y_actual - y_predicted) * (y_actual - y_predicted) 
        ss_tot += (y_actual - y_mean) * (y_actual - y_mean)

    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2

def parse_arguments():
    """
    Parses command-line arguments for metric selection.
    
    Supported arguments:
        --metric: Evaluation metric to use (default: r2)
        --help: Help message
    """
    parser = argparse.ArgumentParser(
        prog="precision.py",
        usage="\"python precision.py [OPTIONS]\" or \"./precision.py [OPTIONS]\"",
        description="Evaluate model accuracy using different metrics.",
        epilog=(
            "Examples:\n"
            "  python precision.py\n"
            "  python precision.py --metric mse\n"
            "  python precision.py --metric rmse\n"
            "  python precision.py --metric r2"
        ), formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        help="Set evaluation metric (default: r2). Supported: mse, rmse, r2"
    )

    args = parser.parse_args()

    if args.metric.lower() not in ['mse', 'rmse', 'r2']:
        parser.error(f"Unsupported metric '{args.metric}'. Supported metrics: mse, rmse, r2.")
    return args.metric.lower()

def main():
    """Main precision program."""
    print("=" * 50)
    print("MODEL ACCURACY EVALUATION PROGRAM (BONUS)")
    print("=" * 50)

    metric=parse_arguments()

    print("\nReading data...")
    mileages, prices = read_dataset(DEFAULT_DATA_FILE)
    print(f"Loaded {len(mileages)} data points.")
    
    print("Reading model parameters...")
    theta0, theta1 = read_trained_data()
    print(f"θ0 = {theta0:.6f}")
    print(f"θ1 = {theta1:.6f}")


    if metric == 'mse':
        mse = calculate_mse(mileages, prices, theta0, theta1)
        print(f"\nMSE (Mean Squared Error): {mse:.2f}")
        print(f"Unit: $²")
        print(f"\nInterpretation:")
        print(f"  • Average squared error: {mse:.2f} $²")
        print(f"  • Smaller value means better model (0 means perfect fit)")
        print(f"  • Larger errors are penalized more heavily when compared to rmse (due to squaring)")
        
    elif metric == 'rmse':
        rmse = calculate_rmse(mileages, prices, theta0, theta1)
        print(f"\nRMSE (Root Mean Squared Error): {rmse:.2f} $")
        print(f"Unit: $")
        print(f"\nInterpretation:")
        print(f"  • Average error: {rmse:.2f} $")
        print(f"  • Predictions deviate by about {rmse:.2f} $ on average")
        print(f"  • Smaller value means better model (0 means perfect fit)")
        
    elif metric == 'r2':
        r2 = calculate_r2(mileages, prices, theta0, theta1)
        percentage = r2 * 100
        print(f"\nR² Score: {r2:.4f}")
        print(f"Unit: None (ratio)")
        print(f"\nInterpretation:")
        print(f"  • Proportion of variance of errors explained (fitted) by the model")
        print(f"  • The model explains {percentage:.2f}% of the variance")
        
        if r2 == 1.0:
            quality = "Perfect"
        elif r2 >= 0.9:
            quality = "Excellent"
        elif r2 >= 0.7:
            quality = "Good"
        elif r2 >= 0.5:
            quality = "Moderate"
        elif r2 >= 0.3:
            quality = "Weak"
        else:
            quality = "Very Weak"
        
        print(f"  • Model quality: {quality}")
        print(f"\nReference:")
        print(f"  1.0  = Perfect fit")
        print(f"  0.9+ = Excellent fit")
        print(f"  0.7+ = Good fit")
        print(f"  0.5+  = Moderate fit")
        print(f"  0.3+  = Weak fit")
        print(f"  0.0+  = Very weak fit")
        print(f"  0.0  = Model is as bad as predicting the mean")
        print(f"  <0   = Model is worse than predicting the mean")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
        