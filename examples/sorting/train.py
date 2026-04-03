"""
Hybrid sorting algorithm optimization.

Optimize the threshold at which quicksort switches to insertion sort,
pivot selection strategy, and other parameters.

Metric: time_ms (lower is better)
"""

import random
import time

# =============================================================================
# PARAMETERS TO OPTIMIZE
# =============================================================================

# Threshold for switching from quicksort to insertion sort
# Small arrays are faster with insertion sort due to lower overhead
INSERTION_THRESHOLD = 16

# Pivot selection strategy: "first", "middle", "median_of_three", "random"
PIVOT_STRATEGY = "median_of_three"

# Partition scheme: "lomuto" or "hoare"
PARTITION_SCHEME = "lomuto"

# Use iterative quicksort (with explicit stack) vs recursive
USE_ITERATIVE = False

# Number of test arrays and their sizes
NUM_ARRAYS = 100
ARRAY_SIZE = 10000

# =============================================================================
# SORTING IMPLEMENTATIONS
# =============================================================================

def insertion_sort(arr, low, high):
    """Sort arr[low:high+1] using insertion sort."""
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def get_pivot_index(arr, low, high):
    """Select pivot index based on strategy."""
    if PIVOT_STRATEGY == "first":
        return low
    elif PIVOT_STRATEGY == "middle":
        return (low + high) // 2
    elif PIVOT_STRATEGY == "random":
        return random.randint(low, high)
    elif PIVOT_STRATEGY == "median_of_three":
        mid = (low + high) // 2
        a, b, c = arr[low], arr[mid], arr[high]
        if a <= b <= c or c <= b <= a:
            return mid
        elif b <= a <= c or c <= a <= b:
            return low
        else:
            return high
    else:
        return low


def partition_lomuto(arr, low, high):
    """Lomuto partition scheme - pivot ends at final position."""
    pivot_idx = get_pivot_index(arr, low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
    pivot = arr[high]
    
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def partition_hoare(arr, low, high):
    """Hoare partition scheme - fewer swaps, pivot not at final position."""
    pivot_idx = get_pivot_index(arr, low, high)
    pivot = arr[pivot_idx]
    
    i = low - 1
    j = high + 1
    
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        
        j -= 1
        while arr[j] > pivot:
            j -= 1
        
        if i >= j:
            return j
        
        arr[i], arr[j] = arr[j], arr[i]


def partition(arr, low, high):
    """Partition array using configured scheme."""
    if PARTITION_SCHEME == "hoare":
        return partition_hoare(arr, low, high)
    else:
        return partition_lomuto(arr, low, high)


def quicksort_recursive(arr, low, high):
    """Recursive quicksort with insertion sort for small subarrays."""
    if high - low + 1 <= INSERTION_THRESHOLD:
        insertion_sort(arr, low, high)
        return
    
    if low < high:
        pi = partition(arr, low, high)
        if PARTITION_SCHEME == "hoare":
            quicksort_recursive(arr, low, pi)
            quicksort_recursive(arr, pi + 1, high)
        else:
            quicksort_recursive(arr, low, pi - 1)
            quicksort_recursive(arr, pi + 1, high)


def quicksort_iterative(arr, low, high):
    """Iterative quicksort using explicit stack."""
    stack = [(low, high)]
    
    while stack:
        low, high = stack.pop()
        
        if high - low + 1 <= INSERTION_THRESHOLD:
            insertion_sort(arr, low, high)
            continue
        
        if low < high:
            pi = partition(arr, low, high)
            
            if PARTITION_SCHEME == "hoare":
                left_low, left_high = low, pi
                right_low, right_high = pi + 1, high
            else:
                left_low, left_high = low, pi - 1
                right_low, right_high = pi + 1, high
            
            # Push larger subarray first (optimization for stack depth)
            if left_high - left_low < right_high - right_low:
                stack.append((right_low, right_high))
                stack.append((left_low, left_high))
            else:
                stack.append((left_low, left_high))
                stack.append((right_low, right_high))


def hybrid_sort(arr):
    """Sort array using hybrid quicksort."""
    if len(arr) <= 1:
        return
    
    if USE_ITERATIVE:
        quicksort_iterative(arr, 0, len(arr) - 1)
    else:
        quicksort_recursive(arr, 0, len(arr) - 1)


# =============================================================================
# BENCHMARK
# =============================================================================

def generate_test_data():
    """Generate test arrays of various patterns."""
    arrays = []
    for i in range(NUM_ARRAYS):
        pattern = i % 5
        if pattern == 0:
            # Random
            arr = [random.randint(0, ARRAY_SIZE * 10) for _ in range(ARRAY_SIZE)]
        elif pattern == 1:
            # Nearly sorted
            arr = list(range(ARRAY_SIZE))
            for _ in range(ARRAY_SIZE // 20):
                i, j = random.randint(0, ARRAY_SIZE - 1), random.randint(0, ARRAY_SIZE - 1)
                arr[i], arr[j] = arr[j], arr[i]
        elif pattern == 2:
            # Reverse sorted
            arr = list(range(ARRAY_SIZE, 0, -1))
        elif pattern == 3:
            # Many duplicates
            arr = [random.randint(0, 100) for _ in range(ARRAY_SIZE)]
        else:
            # Already sorted
            arr = list(range(ARRAY_SIZE))
        arrays.append(arr)
    return arrays


def verify_sorted(arr):
    """Verify array is sorted."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def benchmark():
    """Run sorting benchmark."""
    print(f"Generating {NUM_ARRAYS} test arrays of size {ARRAY_SIZE}...")
    random.seed(42)  # Reproducible
    test_arrays = generate_test_data()
    
    print(f"Configuration:")
    print(f"  INSERTION_THRESHOLD: {INSERTION_THRESHOLD}")
    print(f"  PIVOT_STRATEGY: {PIVOT_STRATEGY}")
    print(f"  PARTITION_SCHEME: {PARTITION_SCHEME}")
    print(f"  USE_ITERATIVE: {USE_ITERATIVE}")
    print()
    
    # Warm up
    warm_up = [random.randint(0, 1000) for _ in range(1000)]
    hybrid_sort(warm_up)
    
    # Benchmark
    start = time.perf_counter()
    
    errors = 0
    for arr in test_arrays:
        arr_copy = arr.copy()
        hybrid_sort(arr_copy)
        if not verify_sorted(arr_copy):
            errors += 1
    
    elapsed = time.perf_counter() - start
    time_ms = elapsed * 1000
    
    # Output metrics
    print("---")
    print(f"time_ms: {time_ms:.3f}")
    print(f"errors: {errors}")
    print(f"arrays_sorted: {NUM_ARRAYS}")
    print(f"total_elements: {NUM_ARRAYS * ARRAY_SIZE}")
    
    if errors > 0:
        print(f"\nERROR: {errors} arrays not sorted correctly!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(benchmark())
