#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_animate.py - Interactive Gradient Descent Visualization
Shows each iteration step by step. Use the space key to move forward.

Usage:
    python train_animate.py                           # Default: 100 iterations
    python train_animate.py --iterations 200          # Number of iterations
    python train_animate.py --learning-rate 0.05      # Learning rate
    python train_animate.py --metric r2               # Metric selection (cost, r2, mse, rmse)
"""

import sys
import argparse
import time
from constants import DEFAULT_DATA_FILE, DEFAULT_LEARNING_RATE, CONVERGENCE_THRESHOLD, LOW_DATA_WARNING_THRESHOLD, ANSI_WARNING, ANSI_RESET
from train import read_dataset, normalize_data, calculate_cost, denormalize_thetas
from precision import calculate_mse, calculate_rmse, calculate_r2


DEFAULT_ANIMATION_ITERATIONS = 1000
LARGE_ITERATION_WARNING_THRESHOLD = 5000
CONVERGENCE_CHECK_INTERVAL = 10


def calculate_precision_metrics(mileages, prices, theta0, theta1):
    """Computes MSE, RMSE and R² on the current model parameters."""
    mse = calculate_mse(mileages, prices, theta0, theta1)
    rmse = mse ** 0.5
    r2 = calculate_r2(mileages, prices, theta0, theta1)
    return mse, rmse, r2


def train_and_collect_iterations(mileages, prices, learning_rate, max_iter, metric_type):
    """
    Trains with gradient descent and stores every iteration.
    
    Returns:
        iterations: List of [theta0, theta1, cost, metric_value, no_improvement_count, last_cost_change]
        training_info: Metadata about convergence checks and early stopping
    """
    m = len(mileages)
    theta0 = 0.0
    theta1 = 0.0

    previous_cost = float('inf')
    no_improvement_count = 0
    patience = 50
    
    print(f"\nTraining started... ({max_iter} iterations)")
    print(f"Metric: {metric_type.upper()}")
    print("=" * 50)

    iterations = []

    for iteration in range(1, max_iter + 1):
        sum_error_theta0 = 0
        sum_error_theta1 = 0
        
        for i in range(m):
            prediction = theta0 + theta1 * mileages[i]
            error = prediction - prices[i]

            sum_error_theta0 += error
            sum_error_theta1 += error * mileages[i]

        temp_theta0 = learning_rate * (sum_error_theta0 / m)
        temp_theta1 = learning_rate * (sum_error_theta1 / m)
        
        theta0 = theta0 - temp_theta0
        theta1 = theta1 - temp_theta1

        current_cost = calculate_cost(mileages, prices, theta0, theta1)

        #Calculate the selected metric to be displayed in the visualization
        if metric_type == 'rmse':
            metric_value = calculate_rmse(mileages, prices, theta0, theta1)
        elif metric_type == 'r2':
            metric_value = calculate_r2(mileages, prices, theta0, theta1)
        else:
            metric_value = calculate_mse(mileages, prices, theta0, theta1)

        last_cost_change = abs(previous_cost - current_cost)

        if iteration % CONVERGENCE_CHECK_INTERVAL == 0 or iteration == max_iter:
            if iteration % 50 == 0:
                 print(f"Iteration {iteration:3d}/{max_iter} recorded... (Cost: {current_cost:.6f}, {metric_type.upper()}: {metric_value:.6f})")

            # Check for convergence
            if last_cost_change < CONVERGENCE_THRESHOLD:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"\nEarly stopping ! Convergence reached at iteration {iteration}.")
                    print(f"Cost change {last_cost_change:.6f} is less than threshold {CONVERGENCE_THRESHOLD}.")
                    break
            else:
                no_improvement_count = 0
            
            previous_cost = current_cost

        # Store this iteration
        iterations.append([
            theta0,
            theta1,
            current_cost,
            metric_value,
            no_improvement_count,
            last_cost_change,
        ])
    
    print("=" * 50)
    print("Training completed!")
    print(f"Recorded a total of {len(iterations)} iterations.\n")

    training_info = {
        'convergence_threshold': CONVERGENCE_THRESHOLD,
        'patience': patience,
        'check_interval': CONVERGENCE_CHECK_INTERVAL,
    }

    return iterations, training_info


def visualize_interactive(mileages, prices, iterations, mile_mean, mile_std, price_mean, price_std, metric_type, training_info):
    """
    Interactive visualization.
    Use the space key to move to the next iteration.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not found.")
        print("Install it with: pip install matplotlib")
        sys.exit(1)

    # Metric metadata (right plot will show REAL metrics on original scale)
    metric_info = {
        'mse': {'name': 'MSE', 'unit': ' (TL²)', 'better': 'lower'},
        'rmse': {'name': 'RMSE', 'unit': ' (TL)', 'better': 'lower'},
        'r2': {'name': 'R² Score', 'unit': '', 'better': 'higher'},
    }

    # metric_type is guaranteed by argparse to be one of the keys
    displayed_metric = metric_type
    current_metric_info = metric_info[displayed_metric]

    # Global state
    state = {'current_iter': 0}

    # Create figure (make left plot wider than right)
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    def update_plot():
        """Updates the chart."""
        iter_idx = state['current_iter']

        if iter_idx >= len(iterations):
            return

        theta0_norm, theta1_norm, cost_norm, _metric_value_norm, no_improve, last_cost_change = iterations[iter_idx]

        # Denormalize theta values
        theta0_real, theta1_real = denormalize_thetas(
            theta0_norm, theta1_norm,
            mile_mean, mile_std,
            price_mean, price_std
        )

        # Precision metrics on original data scale
        mse_real, rmse_real, r2_real = calculate_precision_metrics(
            mileages, prices, theta0_real, theta1_real
        )

        # Build RIGHT plot series on ORIGINAL scale (no normalization)
        if displayed_metric == 'mse':
            current_metric_value = mse_real
            metric_series = []
            for it in iterations[:iter_idx + 1]:
                t0_n, t1_n = it[0], it[1]
                t0_r, t1_r = denormalize_thetas(t0_n, t1_n, mile_mean, mile_std, price_mean, price_std)
                metric_series.append(calculate_mse(mileages, prices, t0_r, t1_r))
        elif displayed_metric == 'rmse':
            current_metric_value = rmse_real
            metric_series = []
            for it in iterations[:iter_idx + 1]:
                t0_n, t1_n = it[0], it[1]
                t0_r, t1_r = denormalize_thetas(t0_n, t1_n, mile_mean, mile_std, price_mean, price_std)
                metric_series.append(calculate_rmse(mileages, prices, t0_r, t1_r))
        else:  # r2
            current_metric_value = r2_real
            metric_series = []
            for it in iterations[:iter_idx + 1]:
                t0_n, t1_n = it[0], it[1]
                t0_r, t1_r = denormalize_thetas(t0_n, t1_n, mile_mean, mile_std, price_mean, price_std)
                metric_series.append(calculate_r2(mileages, prices, t0_r, t1_r))

        # Left chart: data + regression line + predictions
        ax1.clear()

        # Actual data (blue)
        ax1.scatter(mileages, prices, color='blue', label='Actual Data', alpha=0.6, s=50, zorder=3)

        # Predicted values for each data point (green)
        predictions = []
        for mile in mileages:
            pred = theta0_real + theta1_real * mile
            predictions.append(pred)

        ax1.scatter(mileages, predictions, color='green', label='Predicted',
                   alpha=0.6, s=50, zorder=2)

        # Lines between actual and predicted values (showing the error)
        for i in range(len(mileages)):
            ax1.plot([mileages[i], mileages[i]], [prices[i], predictions[i]],
                    color='gray', linestyle='--', alpha=0.3, linewidth=1, zorder=1)

        # Regression line (red)
        min_mile = min(mileages)
        max_mile = max(mileages)
        line_x = [min_mile, max_mile]
        line_y = [theta0_real + theta1_real * min_mile,
                  theta0_real + theta1_real * max_mile]

        ax1.plot(line_x, line_y, color='red', linewidth=2, label='Regression Line', zorder=4)

        ax1.set_xlabel('Mileage (km)', fontsize=11)
        ax1.set_ylabel('Price (TL)', fontsize=11)
        ax1.set_title(f'Iteration {iter_idx}/{len(iterations)-1}', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('auto')

        # Show theta values
        info_text = f'θ₀ = {theta0_real:.2f}\nθ₁ = {theta1_real:.6f}\nCost(norm) = {cost_norm:.6f}'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        precision_text = (
            f'Precision (current iter)\n'
            f'MSE: {mse_real:.2f} TL²\n'
            f'RMSE: {rmse_real:.2f} TL\n'
            f'R²: {r2_real:.4f}'
        )
        ax1.text(0.98, 0.98, precision_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.75))

        # Right chart: metric change over time (REAL scale)
        ax2.clear()

        iters = list(range(len(metric_series)))
        ax2.plot(iters, metric_series, color='green', linewidth=2, marker='o', markersize=3)
        ax2.scatter([iter_idx], [current_metric_value], color='red', s=100, zorder=5, label='Current')

        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel(f'{current_metric_info["name"]}{current_metric_info["unit"]}', fontsize=11)
        ax2.set_title(f'{current_metric_info["name"]} Change', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        # Metric change summary
        if iter_idx > 0:
            initial_value = metric_series[0]
            current_value = current_metric_value

            if displayed_metric == 'r2':
                improvement = (current_value - initial_value) * 100
                improvement_text = f'{current_metric_info["name"]}: {current_value:.4f}\nIncrease: +{improvement:.2f}%'
            else:
                reduction = ((initial_value - current_value) / initial_value) * 100 if initial_value != 0 else 0
                improvement_text = f'{current_metric_info["name"]}: {current_value:.4f}\nReduction: {reduction:.2f}%'

            ax2.text(0.98, 0.98, improvement_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            if displayed_metric == 'r2':
                value_text = f'{current_metric_info["name"]}: {current_metric_value:.4f}'
            else:
                value_text = f'{current_metric_info["name"]}: {current_metric_value:.4f}'

            ax2.text(0.98, 0.98, value_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        # Convergence status box (same rule as train.py)
        patience = training_info['patience']
        threshold = training_info['convergence_threshold']
        converged = no_improve >= patience

        check_interval = training_info['check_interval']

        if last_cost_change is None:
            check_line = f'Last check: N/A (checks run every {check_interval} iters)'
        else:
            check_line = f'Last |Δcost|: {last_cost_change:.8f}'

        checkpoint_pass = (last_cost_change is not None and last_cost_change < threshold)
        checkpoint_text = 'YES' if checkpoint_pass else 'NO'
        converged_text = 'YES' if converged else 'NO'
        convergence_text = (
            f'Converged (patience): {converged_text}\n'
            f'Checkpoint pass: {checkpoint_text}\n'
            f'Rule: |Δcost| < {threshold}\n'
            f'Stable checks: {no_improve}/{patience}\n'
            f'{check_line}'
        )

        conv_color = 'palegreen' if converged else 'mistyrose'
        ax2.text(0.02, 0.98, convergence_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor=conv_color, alpha=0.8))

        fig.tight_layout()
        plt.draw()

    def on_key(event):
        """Keyboard event handler."""
        if event.key == ' ':  # Space key
            if state['current_iter'] < len(iterations) - 1:
                state['current_iter'] += 1
                update_plot()
            else:
                print("\nAll iterations have been displayed!")
        
        elif event.key == 'left':  # Left arrow - back
            if state['current_iter'] > 0:
                state['current_iter'] -= 1
                update_plot()
        
        elif event.key == 'right':  # Right arrow - forward
            if state['current_iter'] < len(iterations) - 1:
                state['current_iter'] += 1
                update_plot()
        
        elif event.key == 'q' or event.key == 'escape':  # Exit
            plt.close()
    
    # Bind the event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show the first iteration
    update_plot()
    
    # Usage instructions
    fig.suptitle(f'Gradient Descent - {current_metric_info["name"]} - [SPACE] Next | [←/→] Back/Next | [Q] Exit', 
                 fontsize=11, fontweight='bold', y=0.98)
    
    print("\n" + "=" * 60)
    print("INTERACTIVE VISUALIZATION")
    print("=" * 60)
    print("Controls:")
    print("  [SPACE]      : Next iteration")
    print("  [← Left]     : Previous iteration")
    print("  [→ Right]    : Next iteration")
    print("  [Q / ESC]    : Exit")
    print(f"\nDisplayed Metric: {current_metric_info['name']}")
    print("=" * 60)
    
    plt.show()


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="train_animate.py",
        usage="\"python train_animate.py [OPTIONS]\" or \"./train_animate.py [OPTIONS]\"",
        description="Interactive gradient descent visualization (step through iterations).",
        epilog=(
            "Examples:\n"
            "  python train_animate.py\n"
            "  python train_animate.py --iterations 200\n"
            "  python train_animate.py --metric r2\n"
            "  python train_animate.py --learning-rate 0.05 --iterations 150 --metric mse\n\n"
            "Controls inside the window:\n"
            "  SPACE    : Next iteration\n"
            "  ← / →    : Back / Forward\n"
            "  Q / ESC  : Exit"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ANIMATION_ITERATIONS,
        help=f"Number of iterations to record (default: {DEFAULT_ANIMATION_ITERATIONS})",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="rmse",
        choices=["mse", "rmse", "r2"],
        help="Metric to visualize (default: rmse)",
    )

    args = parser.parse_args()

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")

    if args.iterations <= 0:
        parser.error("--iterations must be positive.")

    return args.learning_rate, args.iterations, args.metric


def main():
    """
    Main interactive training program function
    """
    print("=" * 60)
    print("GRADIENT DESCENT INTERACTIVE ANIMATION")
    print("=" * 60)
    
    learning_rate, max_iter, metric = parse_arguments()
    
    print(f"\nParameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations: {max_iter}")
    print(f"  Metric: {metric.upper()}")

    if max_iter > LARGE_ITERATION_WARNING_THRESHOLD:
        print(
            f"\nWarning: You requested {max_iter} iterations. "
            "All iterations are recorded in memory for interactive stepping."
        )
        print(
            "This may increase RAM usage and startup time. "
            "For smoother interaction, try a smaller --iterations value (for example 100-2000)."
        )
    
    print("\nReading data...")
    mileages, prices = read_dataset(DEFAULT_DATA_FILE)
    print(f"  Loaded {len(mileages)} data points.")

    if len(mileages) < LOW_DATA_WARNING_THRESHOLD:
        print(
            f"{ANSI_WARNING}Warning: Your dataset is small "
            f"(There should be at least {LOW_DATA_WARNING_THRESHOLD} rows).{ANSI_RESET}"
        )
        print(f"{ANSI_WARNING}Technical note: The model will run, but more data is recommended for robust and generalizable training.{ANSI_RESET}\n")
        time.sleep(3)

    print("\nNormalizing data...")
    mileages_normed, mileage_mean, mile_std_dev = normalize_data(mileages)
    prices_normed, price_mean, price_std_dev = normalize_data(prices)

    # Train and record iterations
    iterations, training_info = train_and_collect_iterations(
        mileages_normed, prices_normed,
        learning_rate, max_iter, metric
    )
    
    # Visualize interactively
    visualize_interactive(
        mileages, prices, iterations,
        mileage_mean, mile_std_dev,
        price_mean, price_std_dev, metric, training_info
    )
    
    print("\nChart closed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
