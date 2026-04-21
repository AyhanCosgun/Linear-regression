#!/usr/bin/env python3
# coding: utf-8

import sys
from constants import DEFAULT_THETA_FILE

def read_trained_data():
    try:
        with open(DEFAULT_THETA_FILE, "r") as file:
            lines = file.readlines()
            if len(lines) < 2:
                print(f"Caution: The file '{DEFAULT_THETA_FILE}' does not contain enough data.")
            else:
                second_line = lines[1].strip()
                if ',' not in second_line:
                    print(f"Caution: The second line of '{DEFAULT_THETA_FILE}' must contain a comma !\n" \
                    "Usage: <first line>   theta0,theta1\n" \
                    "       <second line>     0.5,1.2")
                else:
                    values = second_line.split(',', 1)
                    theta0 = float(values[0])
                    theta1 = float(values[1])
                    return theta0, theta1
    except FileNotFoundError:
        print(f"Error: The file '{DEFAULT_THETA_FILE}' could not be found. The model has not been trained yet. Run \"python train.py\" !")
    except Exception as e:
        print(f"Error while reading '{DEFAULT_THETA_FILE}': {e}")
    print("Using θ0 = 0.0 and θ1 = 0.0 for predictions.")
    return 0.0, 0.0


def main():
    print("="*50)
    print("Prediction for Linear Regression")
    print("="*50)
    theta0, theta1 = read_trained_data()
    try:
        mileage_input = input("\nPlease type the mileage of the car : ")
        mileage = float(mileage_input)
        if mileage < 0:
            print("Caution: Mileage cannot be negative. Please enter a non-negative value !")
            sys.exit(1)
        predicted_price = theta0 + theta1 * mileage
        print("\n" + "=" * 50)
        print(f"Mileage: {mileage:.2f} km")
        print(f"Predicted Price: {predicted_price:.2f} $")
        print("=" * 50)
    except ValueError:
        print("Error: Invalid input for mileage. Please enter a numeric value.")
        sys.exit(1)
    except EOFError:
        print("\nPrediction process ended because no input was provided.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nPrediction process interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
