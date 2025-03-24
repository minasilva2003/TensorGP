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
        fit = old_tf_rmse(tensors[i], target)
        if condition():
            max_fit = fit
            best_ind = i

        times.append((time.time() - start_ind) * 1000.0)
        population[i]['fitness'] = fit

    return population, best_ind


# Different types of function sets
extended_fset = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt',
                 'div', 'exp', 'log', 'warp'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}

if __name__ == "__main__":

    # GP params
    dev = '/gpu:0'  # device to run, write '/cpu_0' to tun on cpu
    gens = 2000  # 50
    pop_size = 100  # 50
    tour_size = 2
    mut_rate = 0.1
    cross_rate = 0.9
    max_tree_dep = 10
    max_init_depth = 10
    elite_size = 1  # 0 to turn off
    runs = 30  # Number of average runs

    # problems
    pagie = "add(div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(x, x), mult(x, x))))), div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(y, y), mult(y, y))))))"
    keijzer11 = "add(mult(x, var), sin(mult(sub(x, scalar(1.0)), sub(var, scalar(1.0)))))"

    problems = ["collatz_numbers", "median", "number_io", "smallest", "sum_of_squares", "wallis_pi", "bouncing_balls", "dice_game", "gcd", "snow_day"]  # Add to run more problems

    for p in problems:

         # Load CSV using Pandas
        df = pd.read_csv(f"cases/{p}_cases.csv")

        # Convert to TensorFlow dataset
   
        dataset = tf.convert_to_tensor(df.to_numpy())

    
        # Domains dimensions
        test_cases = [[dataset.shape[0], dataset.shape[1]]]  # Add more dimensions to test

        for res in test_cases:

            for r in range(runs):
                # seeds = random.randint(0, 0x7fffffff)
                seeds = 39485793482  # reproducibility

                # create engine
                engine = Engine(fitness_func=calc_fit,
                                population_size=pop_size,
                                tournament_size=tour_size,
                                mutation_rate=mut_rate,
                                crossover_rate=cross_rate,
                                max_tree_depth=max_tree_dep,
                                target_dims=res,
                                target=dataset,
                                elitism=elite_size,
                                method='ramped half-and-half',
                                max_init_depth=max_init_depth,
                                objective='minimizing',
                                domain_mode='log',
                                device=dev,
                                stop_criteria='generation',
                                stop_value=gens,
                                effective_dims=2,
                                domain=[-100000, 100000],
                                codomain = [-100000, 100000],  # pagie codomain for the specified [-5, 5 ] range
                                do_final_transform = True,
                                final_transform = [0, 2],
                                operators = normal_set,
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
                                bloat_control="weak",
                                problem_name=p
                                )
                

                # run evolutionary process
                engine.run()
