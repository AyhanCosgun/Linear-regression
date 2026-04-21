#!/usr/bin/env python3
# coding: utf-8

"""Visualizes the training data and the fitted regression line."""

import sys
from constants import DEFAULT_DATA_FILE, ANSI_WARNING, ANSI_RESET, ANSI_ERROR
from train import read_dataset
from predict import read_trained_data

def main():
    
    print("=" * 50)
    print("DATA VISUALIZATION PROGRAM (BONUS)")
    print("=" * 50)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n{ANSI_ERROR}Error: matplotlib library not found.{ANSI_RESET}")
        print(f"{ANSI_WARNING}Install it with: pip install matplotlib{ANSI_RESET}")
        sys.exit(1)
    
    print("\nReading data set...")
    mileages, prices = read_dataset(DEFAULT_DATA_FILE)
    print(f"Loaded {len(mileages)} data points.")

    theta_0, theta_1 = read_trained_data()

    plt.figure(figsize=(15, 9))

    plt.scatter(mileages, prices, color='blue', label='Actual Data', alpha=0.6, s=50, zorder=2)
    if theta_0 == 0.0 and theta_1 == 0.0:
        plt.text(0.5, 0.5, 'The model has not been trained yet !\nRun  \"python train.py\"  to train the model.',
                 transform=plt.gca().transAxes, fontsize=20, color='w', ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='r', alpha=0.3))
    else:
        line_start_x = min(mileages)
        line_end_x = max(mileages)
        line_start_y = theta_0 + theta_1 * line_start_x
        line_end_y = theta_0 + theta_1 * line_end_x

        equation_formula = f"y = ({theta_0:.2f}) + ({theta_1:.2f}) * x"
        plt.plot([line_start_x, line_end_x], [line_start_y, line_end_y],
                  color='red', label=f'Linear Regression Line', linewidth=2, zorder=3)
        plt.plot([],[],linestyle='none' , label=f'{equation_formula}\n')

        # Predicted values for each data point (green)
        predictions = []
        for mile in mileages:
            pred = theta_0+ theta_1 * mile
            predictions.append(pred)
        plt.scatter(mileages, predictions, color='green', label='Predicted', 
                   alpha=0.6, s=50)
        # Lines between actual and predicted values (showing the error)
        for i in range(len(mileages)):
            plt.plot([mileages[i], mileages[i]], [prices[i], predictions[i]], 
                    color='gray', linestyle='--', alpha=0.5, linewidth=1)
        plt.plot([],[], color='gray', linestyle='--' , label=f'errors')
    
    plt.title('Car Price Prediction - Linear Regression', fontsize=14, fontweight='bold')
    plt.xlabel('Mileage (km)', fontsize=12,  fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12,  fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.5)
        
    print("\nDisplaying chart...")
    plt.tight_layout()
    plt.show()
    
    print("Chart closed.")
    

if __name__ == "__main__":
    main()

