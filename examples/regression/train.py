"""
Linear regression optimization.

Optimize gradient descent parameters for fitting a linear model.

Metric: mse (lower is better)
"""

import math
import random
import time

# =============================================================================
# PARAMETERS TO OPTIMIZE
# =============================================================================

# Learning rate
LEARNING_RATE = 0.01

# Number of training iterations
NUM_ITERATIONS = 1000

# Batch size (0 = full batch gradient descent)
BATCH_SIZE = 32

# Regularization strength (L2)
REGULARIZATION = 0.001

# Learning rate schedule: "constant", "linear_decay", "exponential_decay", "step_decay"
LR_SCHEDULE = "constant"

# Momentum (0 = no momentum)
MOMENTUM = 0.0

# Early stopping patience (0 = disabled)
EARLY_STOPPING = 0

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_samples, n_features, noise=0.1):
    """Generate synthetic linear regression data."""
    # True weights
    true_weights = [random.gauss(0, 1) for _ in range(n_features)]
    true_bias = random.gauss(0, 1)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate features
        x = [random.gauss(0, 1) for _ in range(n_features)]
        
        # Generate target with noise
        target = sum(w * xi for w, xi in zip(true_weights, x)) + true_bias
        target += random.gauss(0, noise)
        
        X.append(x)
        y.append(target)
    
    return X, y, true_weights, true_bias


# =============================================================================
# LINEAR REGRESSION
# =============================================================================

def predict(X, weights, bias):
    """Predict using linear model."""
    predictions = []
    for x in X:
        pred = sum(w * xi for w, xi in zip(weights, x)) + bias
        predictions.append(pred)
    return predictions


def mse_loss(y_true, y_pred):
    """Mean squared error."""
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def get_learning_rate(iteration, base_lr):
    """Get learning rate based on schedule."""
    if LR_SCHEDULE == "constant":
        return base_lr
    elif LR_SCHEDULE == "linear_decay":
        decay = 1 - (iteration / NUM_ITERATIONS)
        return base_lr * max(0.01, decay)
    elif LR_SCHEDULE == "exponential_decay":
        return base_lr * (0.99 ** iteration)
    elif LR_SCHEDULE == "step_decay":
        drops = iteration // (NUM_ITERATIONS // 4)
        return base_lr * (0.5 ** drops)
    return base_lr


def train_linear_regression(X_train, y_train, X_val, y_val, n_features):
    """Train linear regression using gradient descent."""
    # Initialize weights
    weights = [random.gauss(0, 0.01) for _ in range(n_features)]
    bias = 0.0
    
    # Momentum velocities
    v_weights = [0.0] * n_features
    v_bias = 0.0
    
    n_samples = len(X_train)
    batch_size = BATCH_SIZE if BATCH_SIZE > 0 else n_samples
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for iteration in range(NUM_ITERATIONS):
        # Get batch
        if batch_size < n_samples:
            indices = random.sample(range(n_samples), batch_size)
            X_batch = [X_train[i] for i in indices]
            y_batch = [y_train[i] for i in indices]
        else:
            X_batch = X_train
            y_batch = y_train
        
        # Forward pass
        predictions = predict(X_batch, weights, bias)
        
        # Compute gradients
        grad_weights = [0.0] * n_features
        grad_bias = 0.0
        
        for x, y_true, y_pred in zip(X_batch, y_batch, predictions):
            error = y_pred - y_true
            for j in range(n_features):
                grad_weights[j] += error * x[j]
            grad_bias += error
        
        # Average gradients
        m = len(X_batch)
        grad_weights = [g / m for g in grad_weights]
        grad_bias /= m
        
        # Add L2 regularization
        for j in range(n_features):
            grad_weights[j] += REGULARIZATION * weights[j]
        
        # Get learning rate
        lr = get_learning_rate(iteration, LEARNING_RATE)
        
        # Update with momentum
        for j in range(n_features):
            v_weights[j] = MOMENTUM * v_weights[j] - lr * grad_weights[j]
            weights[j] += v_weights[j]
        
        v_bias = MOMENTUM * v_bias - lr * grad_bias
        bias += v_bias
        
        # Track losses periodically
        if iteration % 100 == 0 or iteration == NUM_ITERATIONS - 1:
            train_pred = predict(X_train, weights, bias)
            train_loss = mse_loss(y_train, train_pred)
            train_losses.append(train_loss)
            
            val_pred = predict(X_val, weights, bias)
            val_loss = mse_loss(y_val, val_pred)
            val_losses.append(val_loss)
            
            # Early stopping
            if EARLY_STOPPING > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING:
                        break
    
    return weights, bias, train_losses, val_losses


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark():
    """Run regression benchmark."""
    # Configuration
    n_train = 5000
    n_val = 1000
    n_features = 20
    noise = 0.5
    
    print(f"Dataset: {n_train} train, {n_val} val, {n_features} features")
    print(f"Configuration:")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  NUM_ITERATIONS: {NUM_ITERATIONS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  REGULARIZATION: {REGULARIZATION}")
    print(f"  LR_SCHEDULE: {LR_SCHEDULE}")
    print(f"  MOMENTUM: {MOMENTUM}")
    print(f"  EARLY_STOPPING: {EARLY_STOPPING}")
    print()
    
    # Generate data
    random.seed(42)
    X_all, y_all, true_weights, true_bias = generate_data(
        n_train + n_val, n_features, noise
    )
    
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:], y_all[n_train:]
    
    # Train
    start = time.perf_counter()
    weights, bias, train_losses, val_losses = train_linear_regression(
        X_train, y_train, X_val, y_val, n_features
    )
    elapsed = time.perf_counter() - start
    
    # Final evaluation
    val_pred = predict(X_val, weights, bias)
    final_mse = mse_loss(y_val, val_pred)
    
    # Compute R² score
    y_mean = sum(y_val) / len(y_val)
    ss_tot = sum((y - y_mean) ** 2 for y in y_val)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_val, val_pred))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Weight recovery (how close to true weights)
    weight_error = sum((w - tw) ** 2 for w, tw in zip(weights, true_weights))
    weight_error = math.sqrt(weight_error / n_features)
    
    # Output metrics
    print("---")
    print(f"mse: {final_mse:.6f}")
    print(f"r2: {r2:.6f}")
    print(f"weight_error: {weight_error:.6f}")
    print(f"train_time_s: {elapsed:.4f}")
    print(f"final_train_loss: {train_losses[-1]:.6f}")
    print(f"iterations_run: {len(train_losses) * 100}")
    
    return 0


if __name__ == "__main__":
    exit(benchmark())
