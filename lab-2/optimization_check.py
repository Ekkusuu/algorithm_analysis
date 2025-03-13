import time
import random
import matplotlib.pyplot as plt
import heapq
import bisect

# --------------------- UNOPTIMIZED ALGORITHMS ---------------------

def merge_sort(arr, left, right):
    if right - left > 1:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid, right)
        
        merged = []
        i, j = left, mid
        while i < mid and j < right:
            if arr[i] < arr[j]:
                merged.append(arr[i])
                i += 1
            else:
                merged.append(arr[j])
                j += 1
        while i < mid:
            merged.append(arr[i])
            i += 1
        while j < right:
            merged.append(arr[j])
            j += 1

        for index, val in enumerate(merged):
            arr[left + index] = val
    return

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def patience_sort(arr):
    piles = []
    for num in arr:
        placed = False
        for pile in piles:
            if pile[-1] >= num:
                pile.append(num)
                placed = True
                break
        if not placed:
            piles.append([num])

    sorted_arr = []
    while piles:
        smallest = min(piles, key=lambda x: x[-1])
        sorted_arr.append(smallest.pop())
        if not smallest:
            piles.remove(smallest)

    arr[:] = sorted_arr
    return

# --------------------- OPTIMIZED ALGORITHMS ---------------------

def optimized_merge_sort(arr, left, right):
    if right - left <= 10:
        insertion_sort(arr, left, right)
        return
    
    if right - left > 1:
        mid = (left + right) // 2
        optimized_merge_sort(arr, left, mid)
        optimized_merge_sort(arr, mid, right)

        merged = []
        i, j = left, mid
        while i < mid and j < right:
            if arr[i] < arr[j]:
                merged.append(arr[i])
                i += 1
            else:
                merged.append(arr[j])
                j += 1
        while i < mid:
            merged.append(arr[i])
            i += 1
        while j < right:
            merged.append(arr[j])
            j += 1

        for index, val in enumerate(merged):
            arr[left + index] = val
    return

def insertion_sort(arr, left, right):
    for i in range(left + 1, right):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def optimized_heap_sort(arr):
    n = len(arr)
    
    # Iteratively build the heap
    for i in range(n // 2 - 1, -1, -1):
        iterative_heapify(arr, n, i)

    # Extract elements iteratively
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        iterative_heapify(arr, i, 0)

def iterative_heapify(arr, n, i):
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest  # Move down iteratively

def optimized_patience_sort(arr):
    piles = []
    for num in arr:
        idx = bisect.bisect_right([pile[-1] for pile in piles], num)
        if idx < len(piles):
            piles[idx].append(num)
        else:
            piles.append([num])

    heap = [(pile.pop(), i) for i, pile in enumerate(piles)]
    heapq.heapify(heap)
    sorted_index = 0

    while heap:
        smallest, pile_idx = heapq.heappop(heap)
        arr[sorted_index] = smallest
        sorted_index += 1
        if piles[pile_idx]:
            heapq.heappush(heap, (piles[pile_idx].pop(), pile_idx))
    return



# --------------------- TIME COMPLEXITY TESTING ---------------------

def measure_time(sort_function, arr):
    start_time = time.perf_counter()
    sort_function(arr)
    end_time = time.perf_counter()
    return end_time - start_time

def run_experiment():
    sizes = list(range(100, 4001, 100))  # Testing from size 100 to 2000
    unoptimized_times = {"Merge Sort": [], "Heap Sort": [], "Patience Sort": []}
    optimized_times = {"Merge Sort": [], "Heap Sort": [], "Patience Sort": []}

    max_time = 0  # Track maximum time for consistent Y-axis scaling

    for size in sizes:
        test_arr = [random.randint(1, 10000) for _ in range(size)]

        # Measure unoptimized sorting times
        t1 = measure_time(lambda arr: merge_sort(arr, 0, len(arr)), test_arr.copy())
        t2 = measure_time(heap_sort, test_arr.copy())
        t3 = measure_time(patience_sort, test_arr.copy())

        unoptimized_times["Merge Sort"].append(t1)
        unoptimized_times["Heap Sort"].append(t2)
        unoptimized_times["Patience Sort"].append(t3)

        # Measure optimized sorting times
        t4 = measure_time(lambda arr: optimized_merge_sort(arr, 0, len(arr)), test_arr.copy())
        t5 = measure_time(optimized_heap_sort, test_arr.copy())
        t6 = measure_time(optimized_patience_sort, test_arr.copy())

        optimized_times["Merge Sort"].append(t4)
        optimized_times["Heap Sort"].append(t5)
        optimized_times["Patience Sort"].append(t6)

        max_time = max(max_time, t1, t2, t3, t4, t5, t6)  # Update max time

    # Plot results for each algorithm
    for algo in unoptimized_times.keys():
        plt.figure(figsize=(8, 6))
        plt.plot(sizes, unoptimized_times[algo], marker='o', linestyle='-', label=f'Unoptimized {algo}', color='red')
        plt.plot(sizes, optimized_times[algo], marker='o', linestyle='-', label=f'Optimized {algo}', color='green')
        plt.xlabel('Array Size')
        plt.ylabel('Time (seconds)')
        plt.title(f'Comparison of {algo} (Optimized vs. Unoptimized)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max_time * 1.1)  # Keep all graphs within the same scale
        plt.show()

run_experiment()
