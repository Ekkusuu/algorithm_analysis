import tkinter as tk
import random
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SortingVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Sorting Algorithm Visualizer")
        
        # Variables for settings
        self.algorithm = tk.StringVar(value="Quick Sort")
        self.speed = tk.DoubleVar(value=0.05)   # Delay in seconds
        self.size = tk.IntVar(value=50)
        self.include_negatives = tk.BooleanVar(value=False)  # Variable to include negative numbers
        self.include_floats = tk.BooleanVar(value=False)  # Variable to include floating numbers
        
        # Control frame for buttons and sliders
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Algorithm selection
        tk.Label(control_frame, text="Algorithm:").pack(side=tk.LEFT, padx=5)
        alg_options = ["Quick Sort", "Merge Sort", "Heap Sort", "Patience Sort"]
        self.algorithm_menu = tk.OptionMenu(control_frame, self.algorithm, *alg_options)
        self.algorithm_menu.pack(side=tk.LEFT, padx=5)
        
        # Speed slider
        tk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_slider = tk.Scale(control_frame, from_=0, to=2, orient=tk.HORIZONTAL, variable=self.speed,
                                     label="Speed", tickinterval=1)
        self.speed_slider.pack(side=tk.LEFT, padx=5)
        
        # Array size slider
        tk.Label(control_frame, text="Array Size:").pack(side=tk.LEFT, padx=5)
        self.size_slider = tk.Scale(control_frame, from_=10, to=200, orient=tk.HORIZONTAL, 
                                    variable=self.size)
        self.size_slider.pack(side=tk.LEFT, padx=5)
        
        # Option to include negative numbers
        self.negatives_check = tk.Checkbutton(control_frame, text="Include Negative Numbers", variable=self.include_negatives)
        self.negatives_check.pack(side=tk.LEFT, padx=5)
        
        # Option to include floating-point numbers
        self.floats_check = tk.Checkbutton(control_frame, text="Include Floating Numbers", variable=self.include_floats)
        self.floats_check.pack(side=tk.LEFT, padx=5)
        
        # Generate new array button
        self.generate_button = tk.Button(control_frame, text="Generate Array", command=self.generate_array)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        
        # Start sorting button
        self.start_button = tk.Button(control_frame, text="Start Sorting", command=self.start_sorting)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Create the matplotlib Figure
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        
        # Remove axes, ticks, and labels for clean bars
        self.ax.axis('off')  # Hide the axes
        self.ax.set_xticks([])  # Remove x-axis ticks
        self.ax.set_yticks([])  # Remove y-axis ticks
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Initialize data and generator
        self.array = []
        self.sorting_generator = None
        self.running = False
        self.generate_array()
    
    def generate_array(self):
        if self.running:
            # Stop sorting if sorting is running
            self.running = False
            self.sorting_generator = None
            print("Sorting stopped.")
        
        # Generate a new random array with options for negative and floating numbers
        size = self.size.get()
        include_negatives = self.include_negatives.get()
        include_floats = self.include_floats.get()
        
        if include_floats:
            # Generate array with floating-point numbers
            if include_negatives:
                self.array = [round(random.uniform(-100, 100), 2) for _ in range(size)]
            else:
                self.array = [round(random.uniform(0, 100), 2) for _ in range(size)]
        else:
            # Generate array with integers
            if include_negatives:
                self.array = [random.randint(-100, 100) for _ in range(size)]
            else:
                self.array = [random.randint(1, 100) for _ in range(size)]
        
        self.draw_array()
    
    def draw_array(self, highlight_indices=None):
        self.ax.clear()
        
        # Remove axes, ticks, and labels again after clearing
        self.ax.axis('off')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Default color for bars
        bar_colors = ['blue'] * len(self.array)
        # Highlight indices that were recently modified
        if highlight_indices:
            for idx in highlight_indices:
                if 0 <= idx < len(bar_colors):
                    bar_colors[idx] = 'red'
        
        # Draw bars
        self.ax.bar(range(len(self.array)), self.array, color=bar_colors, width=1.0)
        self.canvas.draw()
    
    def start_sorting(self):
        if self.running:
            return  # Prevent starting multiple sorts at once
        self.running = True
        
        # Create a sorting generator based on the selected algorithm
        algo = self.algorithm.get()
        if algo == "Quick Sort":
            self.sorting_generator = quick_sort(self.array, 0, len(self.array)-1)
        elif algo == "Merge Sort":
            self.sorting_generator = merge_sort(self.array, 0, len(self.array))
        elif algo == "Heap Sort":
            self.sorting_generator = heap_sort(self.array)
        elif algo == "Patience Sort":
            self.sorting_generator = patience_sort(self.array)
        
        self.animate()
    
    def animate(self):
        if not self.running:
            return  # Stop animation if sorting is no longer running
        
        try:
            # Get the next state from the sorting generator
            arr_state, highlight = next(self.sorting_generator)
            self.draw_array(highlight_indices=highlight)
            delay = self.map_speed_to_delay(self.speed.get())  # get appropriate delay
            self.master.after(delay, self.animate)
        except StopIteration:
            self.running = False
            self.draw_array()  # Final state of the array
            print("Sorting completed!")
    
    def map_speed_to_delay(self, speed_value):
        """ Map slider values (0, 1, 2) to specific delays. """
        if speed_value == 0:
            return 1  # Fastest (1 ms)
        elif speed_value == 1:
            return 100  # Medium speed (100 ms)
        else:
            return 300  # Slow speed (300 ms)

# ---------------- Sorting Algorithms as Generators ----------------

def quick_sort(arr, low, high):
    if low < high:
        pivot_index = yield from partition(arr, low, high)
        yield from quick_sort(arr, low, pivot_index-1)
        yield from quick_sort(arr, pivot_index+1, high)
    return

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            # Yield the array state after each swap (highlight the swapped indices)
            yield arr, (i, j)
    arr[i+1], arr[high] = arr[high], arr[i+1]
    yield arr, (i+1, high)
    return i+1

def merge_sort(arr, left, right):
    if right - left > 1:
        mid = (left + right) // 2
        yield from merge_sort(arr, left, mid)
        yield from merge_sort(arr, mid, right)
        
        merged = []
        i, j = left, mid
        # Merge the two sorted halves
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
        
        # Update the original array and yield the state after each element is placed
        for index, val in enumerate(merged):
            arr[left + index] = val
            yield arr, (left + index,)
    return

def heap_sort(arr):
    n = len(arr)
    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(arr, n, i)
    # Extract elements one by one
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        yield arr, (0, i)
        yield from heapify(arr, i, 0)
    return

def heapify(arr, n, i):
    largest = i
    left = 2*i + 1
    right = 2*i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        yield arr, (i, largest)
        yield from heapify(arr, n, largest)
    return

def patience_sort(arr):
    piles = []
    
    for num in arr:
        placed = False
        for pile in piles:
            if pile[-1] >= num:
                pile.append(num)
                placed = True
                yield arr, [arr.index(num)]  # Highlight the inserted index
                break
        if not placed:
            piles.append([num])
            yield arr, [arr.index(num)]  # Highlight the inserted index
    
    # Merging phase - animate placing elements from piles into the sorted array
    sorted_arr = []
    while piles:
        smallest = min(piles, key=lambda x: x[-1])
        sorted_arr.append(smallest.pop())
        
        if not smallest:
            piles.remove(smallest)
        
        # Update the original array and yield the state with a single smooth progression
        for i in range(len(sorted_arr)):
            arr[i] = sorted_arr[i]
        
        yield arr, range(len(sorted_arr))  # Highlight the updated indices all at once

    # Final state
    arr[:] = sorted_arr
    return

# ---------------- Main Program ----------------

def main():
    root = tk.Tk()
    app = SortingVisualizer(root)
    root.mainloop()

if __name__ == '__main__':
    main()
