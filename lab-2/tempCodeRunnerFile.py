def plot_time_complexity(self):
    """ Analyze and plot the time complexity of the selected sorting algorithm with improved visual differentiation """
    algo = self.algorithm.get()
    include_negatives = self.include_negatives.get()
    include_floats = self.include_floats.get()
    
    sizes = list(range(10, 1001, 10))  # Testing with array sizes from 10 to 1000
    times = []  # Stores times for the selected algorithm

    for size in sizes:
        run_times = []  # Store multiple run times for this array size

        for _ in range(5):  # Run each size 5 times to smooth out anomalies
            # Generate a random array with chosen settings
            if include_floats:
                array = [round(random.uniform(-100, 100), 2) if include_negatives 
                         else round(random.uniform(0, 100), 2) for _ in range(size)]
            else:
                array = [random.randint(-100, 100) if include_negatives 
                         else random.randint(1, 100) for _ in range(size)]
            
            gc.disable()  # Disable garbage collection to avoid random slowdowns
            start_time = time.perf_counter()  # High-precision timing

            # Run sorting algorithm
            if algo == "Quick Sort":
                list(quick_sort(array, 0, len(array) - 1))
            elif algo == "Merge Sort":
                list(merge_sort(array, 0, len(array)))
            elif algo == "Heap Sort":
                list(heap_sort(array))
            elif algo == "Patience Sort":
                list(patience_sort(array))

            end_time = time.perf_counter()
            gc.enable()  # Re-enable garbage collection

            elapsed_time = end_time - start_time  # Time in seconds
            run_times.append(elapsed_time)

        # Take the **average** of multiple runs to smooth out spikes
        avg_time = sum(run_times) / len(run_times)
        times.append(round(avg_time, 8))  # Keep high precision

    # Determine the max time recorded (for scaling Y-axis dynamically)
    max_time = max(times) * 1.1  # Add 10% margin

    # Color coding for different algorithms
    colors = {
        "Quick Sort": "blue",
        "Merge Sort": "green",
        "Heap Sort": "red",
        "Patience Sort": "purple"
    }

    # Plot the graph for the selected algorithm
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, marker='o', linestyle='-', color=colors.get(algo, "black"), label=f'{algo} Time Complexity')
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'{algo} Time Complexity Analysis (Visually Distinct)')
    plt.ylim(0, max_time)  # Set dynamic Y-limit based on max time
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_all_algorithms_time_complexity(self):
    """ Plot the time complexity for all sorting algorithms in one graph """
    include_negatives = self.include_negatives.get()
    include_floats = self.include_floats.get()
    
    sizes = list(range(10, 1001, 10))  # Testing with array sizes from 10 to 1000
    all_algorithms_times = {  # Dictionary to store times for all algorithms
        "Quick Sort": [],
        "Merge Sort": [],
        "Heap Sort": [],
        "Patience Sort": []
    }

    for size in sizes:
        # For each size, run each algorithm and collect times
        for algo_name in all_algorithms_times:
            run_times = []  # Store multiple run times for this array size

            for _ in range(5):  # Run each size 5 times to smooth out anomalies
                # Generate a random array with chosen settings
                if include_floats:
                    array = [round(random.uniform(-100, 100), 2) if include_negatives 
                             else round(random.uniform(0, 100), 2) for _ in range(size)]
                else:
                    array = [random.randint(-100, 100) if include_negatives 
                             else random.randint(1, 100) for _ in range(size)]

                gc.disable()  # Disable garbage collection to avoid random slowdowns
                start_time = time.perf_counter()

                # Run the current algorithm
                if algo_name == "Quick Sort":
                    list(quick_sort(array, 0, len(array) - 1))
                elif algo_name == "Merge Sort":
                    list(merge_sort(array, 0, len(array)))
                elif algo_name == "Heap Sort":
                    list(heap_sort(array))
                elif algo_name == "Patience Sort":
                    list(patience_sort(array))

                end_time = time.perf_counter()
                gc.enable()  # Re-enable garbage collection

                elapsed_time = end_time - start_time  # Time in seconds
                run_times.append(elapsed_time)

            # Take the **average** of multiple runs to smooth out spikes
            avg_time = sum(run_times) / len(run_times)
            all_algorithms_times[algo_name].append(round(avg_time, 8))  # Keep high precision

    # Determine the max time recorded (for scaling Y-axis dynamically)
    max_time = max(max(all_algorithms_times["Quick Sort"]),
                   max(all_algorithms_times["Merge Sort"]),
                   max(all_algorithms_times["Heap Sort"]),
                   max(all_algorithms_times["Patience Sort"])) * 1.1  # Add 10% margin

    # Color coding for different algorithms
    colors = {
        "Quick Sort": "blue",
        "Merge Sort": "green",
        "Heap Sort": "red",
        "Patience Sort": "purple"
    }

    # Plot the graph for all algorithms
    plt.figure(figsize=(8, 6))
    for algo_name in all_algorithms_times:
        plt.plot(sizes, all_algorithms_times[algo_name], marker='o', linestyle='-', color=colors[algo_name], label=f'{algo_name}')
    
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title('All Sorting Algorithms Time Complexity Comparison')
    plt.ylim(0, max_time)  # Dynamic Y-limit for clarity
    plt.grid(True)
    plt.legend()
    plt.show()

# Create the Tkinter interface and add the new button
root = tk.Tk()

# Add other elements such as sliders, labels, etc. 

# Add the button to plot time complexity for all algorithms
plot_all_button = tk.Button(root, text="Compare All Algorithms", command=lambda: plot_all_algorithms_time_complexity(root))
plot_all_button.pack()

# Start the Tkinter event loop
root.mainloop()