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
from constants import DEFAULT_DATA_FILE, DEFAULT_LEARNING_RATE
from train import read_dataset, normalize_data, calculate_cost, denormalize_thetas


DEFAULT_ANIMATION_ITERATIONS = 1000
LARGE_ITERATION_WARNING_THRESHOLD = 5000
CONVERGENCE_CHECK_INTERVAL = 10


def calculate_mse(mileages, prices, theta0, theta1):
    """Computes MSE (Mean Squared Error)."""
    m = len(mileages)
    total_squared_error = 0
    
    for i in range(m):
        prediction = theta0 + theta1 * mileages[i]
        error = prices[i] - prediction
        total_squared_error += error * error
    
    mse = total_squared_error / m
    return mse


def calculate_rmse(mileages, prices, theta0, theta1):
    """Computes RMSE (Root Mean Squared Error)."""
    mse = calculate_mse(mileages, prices, theta0, theta1)
    rmse = mse ** 0.5
    return rmse


def calculate_r2(mileages, prices, theta0, theta1):
    """Computes R² (Coefficient of Determination)."""
    m = len(mileages)
    
    # Compute the mean
    total_price = 0
    for price in prices:
        total_price += price
    mean_price = total_price / m
    
    # Compute SS_res
    ss_res = 0
    for i in range(m):
        prediction = theta0 + theta1 * mileages[i]
        error = prices[i] - prediction
        ss_res += error * error
    
    # Compute SS_tot
    ss_tot = 0
    for price in prices:
        deviation = price - mean_price
        ss_tot += deviation * deviation
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_metric(mileages, prices, theta0, theta1, metric_type):
    """Computes the selected metric."""
    if metric_type == 'cost':
        return calculate_cost(mileages, prices, theta0, theta1)
    elif metric_type == 'mse':
        return calculate_mse(mileages, prices, theta0, theta1)
    elif metric_type == 'rmse':
        return calculate_rmse(mileages, prices, theta0, theta1)
    elif metric_type == 'r2':
        return calculate_r2(mileages, prices, theta0, theta1)
    else:
        return calculate_cost(mileages, prices, theta0, theta1)


def calculate_precision_metrics(mileages, prices, theta0, theta1):
    """Computes MSE, RMSE and R² on the current model parameters."""
    mse = calculate_mse(mileages, prices, theta0, theta1)
    rmse = mse ** 0.5
    r2 = calculate_r2(mileages, prices, theta0, theta1)
    return mse, rmse, r2


def train_and_collect_iterations(mileages_norm, prices_norm, learning_rate, max_iterations, metric_type):
    """
    Trains with gradient descent and stores every iteration.
    
    Returns:
        iterations: List of [theta0, theta1, cost, metric_value, no_improvement_count, last_cost_change]
        training_info: Metadata about convergence checks and early stopping
    """
    m = len(mileages_norm)
    theta0 = 0.0
    theta1 = 0.0

    # Keep convergence logic consistent with train.py
    previous_cost = float('inf')
    convergence_threshold = 0.01
    no_improvement_count = 0
    patience = 10
    
    iterations = []
    
    print(f"\nTraining started... ({max_iterations} iterations)")
    print(f"Metric: {metric_type.upper()}")
    print("=" * 50)
    
    for iteration in range(max_iterations):
        # Compute gradients
        sum_error_theta0 = 0
        sum_error_theta1 = 0
        
        for i in range(m):
            prediction = theta0 + theta1 * mileages_norm[i]
            error = prediction - prices_norm[i]
            sum_error_theta0 += error
            sum_error_theta1 += error * mileages_norm[i]
        
        # Update theta values
        temp_theta0 = learning_rate * (sum_error_theta0 / m)
        temp_theta1 = learning_rate * (sum_error_theta1 / m)
        
        theta0 = theta0 - temp_theta0
        theta1 = theta1 - temp_theta1

        # Compute values after update (same evaluation point as train.py)
        current_cost = calculate_cost(mileages_norm, prices_norm, theta0, theta1)
        metric_value = calculate_metric(mileages_norm, prices_norm, theta0, theta1, metric_type)

        # Convergence is checked every CONVERGENCE_CHECK_INTERVAL iterations (and on the last iteration)
        last_cost_change = None
        is_checkpoint = (iteration % CONVERGENCE_CHECK_INTERVAL == 0 or iteration == max_iterations - 1)
        if is_checkpoint:
            last_cost_change = abs(previous_cost - current_cost)
            if last_cost_change < convergence_threshold:
                no_improvement_count += 1
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
        
        # Show progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration:3d}/{max_iterations} recorded... (Cost: {current_cost:.6f}, {metric_type.upper()}: {metric_value:.6f})")

        # Early stop condition kept consistent with train.py
        if no_improvement_count >= patience:
            print(f"\nEarly stopping! Iteration {iteration} - convergence reached.")
            if last_cost_change is not None:
                print(f"Cost change: {last_cost_change:.8f} < {convergence_threshold}")
            break
    
    print("=" * 50)
    print(f"Recorded a total of {len(iterations)} iterations.\n")

    training_info = {
        'convergence_threshold': convergence_threshold,
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
    
    # Metric metadata
    metric_info = {
        'cost': {'name': 'Cost Function', 'unit': ' (normalized)', 'better': 'lower'},
        'mse': {'name': 'MSE', 'unit': ' (normalize)', 'better': 'lower'},
        'rmse': {'name': 'RMSE', 'unit': ' (normalize)', 'better': 'lower'},
        'r2': {'name': 'R² Score', 'unit': '', 'better': 'higher'}
    }
    
    current_metric_info = metric_info.get(metric_type, metric_info['cost'])
    
    # Global state
    state = {'current_iter': 0}
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def update_plot():
        """Updates the chart."""
        iter_idx = state['current_iter']
        
        if iter_idx >= len(iterations):
            return
        
        theta0_norm, theta1_norm, cost, metric_value, no_improve, last_cost_change = iterations[iter_idx]
        
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
        
        # Show theta values
        info_text = f'θ₀ = {theta0_real:.2f}\nθ₁ = {theta1_real:.6f}\nCost(norm) = {cost:.6f}'
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
        
        # Right chart: metric change over time
        ax2.clear()
        
        # Plot metrics so far
        metrics = [it[3] for it in iterations[:iter_idx+1]]
        iters = list(range(len(metrics)))
        
        ax2.plot(iters, metrics, color='green', linewidth=2, marker='o', markersize=3)
        ax2.scatter([iter_idx], [metric_value], color='red', s=100, zorder=5, label='Current')
        
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel(f'{current_metric_info["name"]}{current_metric_info["unit"]}', fontsize=11)
        ax2.set_title(f'{current_metric_info["name"]} Change', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Metric change summary
        if iter_idx > 0:
            initial_value = iterations[0][3]
            current_value = metric_value
            
            if metric_type == 'r2':
                # Show increase for R²
                improvement = (current_value - initial_value) * 100
                improvement_text = f'{current_metric_info["name"]}: {current_value:.4f}\nIncrease: +{improvement:.2f}%'
                box_color = 'lightgreen'
            else:
                # Show reduction for Cost, MSE, RMSE
                reduction = ((initial_value - current_value) / initial_value) * 100 if initial_value != 0 else 0
                improvement_text = f'{current_metric_info["name"]}: {current_value:.6f}\nReduction: {reduction:.2f}%'
                box_color = 'lightgreen'
            
            ax2.text(0.98, 0.98, improvement_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7))
        else:
            # On the first iteration, show only the value
            value_text = f'{current_metric_info["name"]}: {metric_value:.6f}'
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
    learning_rate = DEFAULT_LEARNING_RATE
    iterations = DEFAULT_ANIMATION_ITERATIONS
    metric = 'cost'
    
    args = sys.argv[1:]
    i = 0
    
    while i < len(args):
        arg = args[i]
        
        if arg == "--help" or arg == "-h":
            print("Usage: python train_animate.py [OPTIONS]")
            print("\nOptions:")
            print(f"  --learning-rate VALUE   Learning rate (default: {DEFAULT_LEARNING_RATE})")
            print(f"  --iterations VALUE      Number of iterations (default: {DEFAULT_ANIMATION_ITERATIONS})")
            print("  --metric VALUE          Metric to visualize (default: cost)")
            print("                          Options: cost, mse, rmse, r2")
            print("  --help, -h              Show this help message")
            print("\nMetrics:")
            print("  cost - Cost Function (gradient descent optimization objective)")
            print("  mse  - Mean Squared Error")
            print("  rmse - Root Mean Squared Error")
            print("  r2   - R² Score (Coefficient of Determination, between 0 and 1)")
            print("\nControls:")
            print("  SPACE    : Next iteration")
            print("  ← / →    : Back / Forward")
            print("  Q / ESC  : Exit")
            print("\nExamples:")
            print("  python train_animate.py")
            print("  python train_animate.py --iterations 200")
            print("  python train_animate.py --metric r2")
            print("  python train_animate.py --learning-rate 0.05 --iterations 150 --metric mse")
            sys.exit(0)
        
        elif arg == "--learning-rate":
            if i + 1 < len(args):
                try:
                    learning_rate = float(args[i + 1])
                    if learning_rate <= 0:
                        print("Error: Learning rate must be positive.")
                        sys.exit(1)
                    i += 2
                except ValueError:
                    print(f"Error: Invalid learning rate: {args[i + 1]}")
                    sys.exit(1)
            else:
                print("Error: No value provided for --learning-rate.")
                sys.exit(1)
        
        elif arg == "--iterations":
            if i + 1 < len(args):
                try:
                    iterations = int(args[i + 1])
                    if iterations <= 0:
                        print("Error: Iteration count must be positive.")
                        sys.exit(1)
                    i += 2
                except ValueError:
                    print(f"Error: Invalid iteration count: {args[i + 1]}")
                    sys.exit(1)
            else:
                print("Error: No value provided for --iterations.")
                sys.exit(1)
        
        elif arg == "--metric":
            if i + 1 < len(args):
                metric = args[i + 1].lower()
                if metric not in ['cost', 'mse', 'rmse', 'r2']:
                    print(f"Error: Invalid metric: {metric}")
                    print("Valid metrics: cost, mse, rmse, r2")
                    sys.exit(1)
                i += 2
            else:
                print("Error: No value provided for --metric.")
                sys.exit(1)
        
        else:
            print(f"Error: Unknown argument: {arg}")
            print("For help: python train_animate.py --help")
            sys.exit(1)
    
    return learning_rate, iterations, metric


def main():
    """Main program."""
    print("=" * 60)
    print("GRADIENT DESCENT INTERACTIVE ANIMATION")
    print("=" * 60)
    
    # Parse arguments
    learning_rate, max_iterations, metric = parse_arguments()
    
    print(f"\nParameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations: {max_iterations}")
    print(f"  Metric: {metric.upper()}")

    if max_iterations > LARGE_ITERATION_WARNING_THRESHOLD:
        print(
            f"\nWarning: You requested {max_iterations} iterations. "
            "All iterations are recorded in memory for interactive stepping."
        )
        print(
            "This may increase RAM usage and startup time. "
            "For smoother interaction, try a smaller --iterations value (for example 100-2000)."
        )
    
    # Read data
    print("\nReading data...")
    mileages, prices = read_dataset(DEFAULT_DATA_FILE)
    print(f"  Loaded {len(mileages)} data points.")
    
    # Normalize data
    print("\nNormalizing data...")
    mileages_norm, mile_mean, mile_std = normalize_data(mileages)
    prices_norm, price_mean, price_std = normalize_data(prices)
    
    # Train and record iterations
    iterations, training_info = train_and_collect_iterations(
        mileages_norm, prices_norm, 
        learning_rate, max_iterations, metric
    )
    
    # Visualize interactively
    visualize_interactive(
        mileages, prices, iterations,
        mile_mean, mile_std,
        price_mean, price_std, metric, training_info
    )
    
    print("\nChart closed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
