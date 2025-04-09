from tensorgp.engine import *
import tensorflow as tf
import torch 
import pandas as pd

# Fitness function to calculate RMSE from target (Pagie Polynomial)
def calc_fit(**kwargs):
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    _stf = kwargs.get('stf')
    target = kwargs.get('target')

    fn = f_path + "gen_" + str(generation).zfill(5)
    fitness = []
    times = []
    best_ind = 0

    # set objective function according to min/max
    fit = 0
    condition = lambda: (fit < max_fit)  # minimizing
    max_fit = float('inf')

    for i in range(len(tensors)):

        start_ind = time.time()
        fit = tensor_rmse(tensors[i], target)
        if condition():
            max_fit = fit
            best_ind = i

        times.append((time.time() - start_ind) * 1000.0)
        population[i]['fitness'] = fit

    return population, best_ind

# Custom var_func to map terminal variables to columns of the input matrix
def custom_var_func(dimensions, column_index):
    return torch.tensor(X[:, column_index].numpy(), dtype=torch.float32)

# Different types of function sets
extended_fset = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt',
                 'div', 'exp', 'log', 'warp'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}

my_set = {"add", "sub", "mult", "div", "abs", "pow", "max", "min"}

if __name__ == "__main__":

    # GP params
    dev = '/gpu:0'  # device to run, write '/cpu_0' to tun on cpu
    gens = 20  # 50
    pop_size = 100  # 50
    tour_size = 5
    mut_rate = 0.9
    cross_rate = 0.5
    max_tree_dep = 12
    max_init_depth=6
    elite_size = 1  # 0 to turn off
    runs = 1 # Number of average runs

    # problems
    pagie = "add(div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(x, x), mult(x, x))))), div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(y, y), mult(y, y))))))"
    keijzer11 = "add(mult(x, var), sin(mult(sub(x, scalar(1.0)), sub(var, scalar(1.0)))))"

    problems = ["collatz_numbers", "median", "number_io", "smallest", "sum_of_squares", "wallis_pi", "bouncing_balls", "dice_game", "gcd", "snow_day"]  # Add to run more problems
   
    for p in problems:

        # Load CSV using Pandas
        df = pd.read_csv(f"cases/{p}_cases.csv")

        # Convert to TensorFlow dataset
        # Convert to PyTorch tensors
        X = torch.tensor(df.iloc[:, :-1].to_numpy(), dtype=torch.float32)
        Y = torch.tensor(df.iloc[:, -1].to_numpy(), dtype=torch.float32)

        # Ensure X always has at least two dimensions
        # Ensure X always has at least two dimensions
        if X.shape[1] == 1:  # If X has only one column
            # Add a column of zeros to X using PyTorch
            zeros_column = torch.zeros([X.shape[0], 1], dtype=X.dtype)  # Create a column of zeros
            X = torch.cat([X, zeros_column], dim=1)  # Concatenate the zeros column to X along the second axis (columns)

        # Domains dimensions
        test_cases = [[Y.shape[0],1]]  # Add more dimensions to test

        for res in test_cases:

            for r in range(runs):
                seeds = random.randint(0, 0x7fffffff)
                #seeds = 39485793482  # reproducibility
                print(dir(Experiment))
                # create engine
                engine = Engine(fitness_func=calc_fit,
                                function_set=Function_Set(my_set,8),
                                population_size=pop_size,
                                tournament_size=tour_size,
                                mutation_rate=mut_rate,
                                crossover_rate=cross_rate,
                                max_tree_depth=max_tree_dep,
                                min_tree_depth=-1,
                                min_init_depth=1,
                                max_init_depth=6,
                                var_func=custom_var_func,
                                effective_dims=X.shape[1],
                                target_dims=res,
                                target=Y,
                                elitism=elite_size,
                                method='ramped half-and-half',

                                objective='minimizing',
                                domain_mode='clip',
                                device=dev,
                                stop_criteria='generation',
                                stop_value=gens,
                                domain=[-100000, 100000],
                                codomain = [-100000, 100000],  # pagie codomain for the specified [-5, 5 ] range
                                do_final_transform = False,
                                final_transform = [0, 2],
                                
                               # mutations
                               max_retries=20,
                               mutation_funcs=[Engine.point_mutation, Engine.subtree_mutation, Engine.insert_mutation, Engine.delete_mutation],
                               mutation_probs=[0.25, 0.3, 0.2, 0.25],
                               min_subtree_dep=None,
                        	      max_subtree_dep=None,
                        
                                operators = my_set,
                                seed = seeds,
                                save_to_file = 2000,
                                save_to_file_image = 2000,
                                save_to_file_log = 2000,
                                save_graphics = False,
                                show_graphics = False,
                                save_image_best = False,
                                save_image_pop = False,
                                save_log = True,
                                write_engine_state = True,
                                read_init_pop_from_file = None,
                        # bloat
                        bloat_control='weak',
                        bloat_mode='depth',
                        dynamic_limit=5,
                        min_overall_size=1,
                        max_overall_size=max_tree_dep,
                                )
                
                # run evolutionary process
                engine.run()