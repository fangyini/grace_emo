import matplotlib.pyplot as plt
import numpy as np
import argparse
# Step 1: Read the file and extract data
parser = argparse.ArgumentParser()
parser.add_argument("--motor_file", type=str, default='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/motor_latency/grace_motor_test2.txt')
args = parser.parse_args()
file_path = args.motor_file

number_of_motors = []
total_time = []
error_count = 0
total_blocks = 0

with open(file_path, 'r') as file:
    lines = file.readlines()
    for i in range(len(lines)):
        if "Number of motors:" in lines[i]:
            motors = int(lines[i].split(":")[1].strip())
            number_of_motors.append(motors)
        
        if "Total time:" in lines[i]:
            time = float(lines[i].split(":")[1].strip())
            total_time.append(time)
            # Increment total_blocks only if time is valid (>= 0)
            if time >= 0:
                total_blocks += 1
        
        if "Motor error:" in lines[i]:
            error_count += 1
            number_of_motors.pop()
            total_time.pop()

# Step 2: Calculate error rate
error_rate = error_count / total_blocks if total_blocks > 0 else 0

# Step 3: Prepare data for plotting
mean_times = []
std_times = []
unique_motors = sorted(set(number_of_motors))

for motors in unique_motors:
    # Filter times for the current number of motors and exclude negative total times
    times = [total_time[i] for i in range(len(number_of_motors)) 
             if number_of_motors[i] == motors and total_time[i] >= 0]
    
    # Only calculate mean and std if there are valid times
    if times:
        mean_times.append(np.mean(times))
        std_times.append(np.std(times))
    else:
        mean_times.append(0)  # or np.nan if you prefer to skip plotting
        std_times.append(0)    # or np.nan if you prefer to skip plotting

# Step 4: Create a bar chart
plt.bar(unique_motors, mean_times, yerr=std_times, capsize=5)
plt.xlabel('Number of Motors')
plt.ylabel('Total Time (seconds)')
plt.title('Mean Total Time vs. Number of Motors')
plt.xticks(unique_motors)
plt.grid(axis='y')

# Show the plot
plt.show()

# Print the error rate
print(f"Error Rate: {error_rate:.2%}")
