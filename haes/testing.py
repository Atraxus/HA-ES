from tabrepo import load_repository, EvaluationRepository
import os

def check_files_for_numbers(directory, numbers):
    # Create a dictionary to keep track of which numbers have been found
    found = {str(number): [] for number in numbers}

    # List all files in the given directory
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Directory '{directory}' does not exist.")
        return
    except PermissionError:
        print(f"Permission denied to access '{directory}'.")
        return

    # Check each file if it contains any of the numbers
    for file in files:
        for number in numbers:
            if str(number) in file:
                found[str(number)].append(file)

    # Print out files that contain the numbers
    for number in numbers:
        if found[str(number)]:
            print(f"#{len(found[str(number)])} Files containing {number}: {', '.join(found[str(number)])}")
        else:
            print(f"No files found containing {number}")


context_name = "D244_F3_C1530_100"
repo: EvaluationRepository = load_repository(context_name, cache=True)
datasets = repo.datasets()
print(f"Datasets: {datasets}")

task_numbers = []
for dataset in datasets:
    tid = repo.dataset_to_tid(dataset)
    print(tid)
    print(repo.tid_to_dataset(tid))
    #print(f"{dataset} tid: {tid}")
    task_numbers.append("_" + str(tid) + "_0")
    
check_files_for_numbers("results/seed_1/", task_numbers)