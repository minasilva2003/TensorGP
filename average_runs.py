import pandas as pd
import os

problems = ["collatz_numbers", "median", "number_io", "smallest", "sum_of_squares", "wallis_pi", "bouncing_balls", "dice_game", "gcd", "snow_day"]  # Add to run more problems

def analyze_fitness(base_directory, problem_name):
    folders = [f for f in os.listdir(base_directory) if f.startswith(problem_name)]
    folders = folders[:30]  # Limit to first 30 folders
    print(f"{problem_name} has {len(folders)} folders")
    
    for folder in folders:
        gen_file_path = os.path.join(base_directory, folder, "logs", "generations", "gen02000.csv")
        stats_file_path = os.path.join(base_directory, folder, "my_stats.csv")
        
        try:
            df = pd.read_csv(gen_file_path)  # Read the CSV file
            if 'fitness' not in df.columns:
                print(f"Warning: 'fitness' column not found in {gen_file_path}")
                continue
            
            min_fitness = df['fitness'].min()  # Get min fitness
            count_below_threshold = (df['fitness'] < 0.1).sum()  # Count values below 0.1
            
            # Create a new DataFrame with the results
            stats_df = pd.DataFrame([[min_fitness, count_below_threshold]], columns=["min_fitness", "num_correct"])
            stats_df["min_fitness"] = stats_df["min_fitness"].round(2)  # Round min_fitness to two decimals
            stats_df.to_csv(stats_file_path, index=False)  # Save to CSV
            
        except FileNotFoundError:
            print(f"Warning: {gen_file_path} not found")
        except Exception as e:
            print(f"Error processing {gen_file_path}: {e}")


def average_problem_stats(parent_directory, problems):
  
    results = []
    
    for problem in problems:
        
        folders = [f for f in os.listdir(parent_directory+"/"+problem) if f.startswith(problem)]

        folders = folders[:30]

        stat_files = [os.path.join(parent_directory+"/"+problem, folder, "my_stats.csv") for folder in folders]
        
        print(f"Processing {problem}... with {len(stat_files)} files")

        all_stats = []
        for stat_file in stat_files:
            try:
                df = pd.read_csv(stat_file)
                all_stats.append(df)
            except FileNotFoundError:
                print(f"Warning: {stat_file} not found")
                continue
        
        if all_stats:
            concatenated_stats = pd.concat(all_stats)
            avg_stats = concatenated_stats.mean().to_frame().T  # Compute average row
            std_stats = concatenated_stats.std().to_frame().T.round(2)  # Compute standard deviation row and round to two decimals
            
            # Round specific columns
            avg_stats["min_fitness"] = avg_stats["min_fitness"].round(2)  # Round min_fitness to two decimals
            avg_stats["num_correct"] = avg_stats["num_correct"].round(0)  # Round num_correct to the nearest unit
            
            # Add problem name
            avg_stats.insert(0, "problem", problem)
            std_stats.insert(0, "problem", problem)
            
            # Rename columns for standard deviation
            std_stats.columns = [f"{col}_std" if col != "problem" else col for col in std_stats.columns]
            
            # Combine average and standard deviation into one DataFrame
            combined_stats = pd.concat([avg_stats, std_stats], axis=1)
            results.append(combined_stats)
    
    if results:
        final_df = pd.concat(results, ignore_index=True)  # Combine all results
        final_df.to_csv("summary_stats.csv", index=False)



# Example usage

for p in problems:
    base_directory = "tensorgp_runs"+"/"+p
    analyze_fitness(base_directory, p)


average_problem_stats("tensorgp_runs", problems)