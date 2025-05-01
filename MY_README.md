# Symbolic Regression Experiments

## === 1. Installation

Follow the **Installation** section from the main `README.md` file to choose your preferred installation method and set up the environment.

---

## === 2. Running Symbolic Regression

1. Run the symbolic regression example script:
   ```bash
   python example_symreg.py
   ```

2. All experiment parameters (e.g., function set, population size, etc.) are defined in `example_symreg.py`.

3. This version of the script includes:
   - A new function set: `my_set`
   - A custom function: `custom_var_func`
     - This function is used by the engine to correctly extract variables (`x1`, `x2`, `x3`, ...) and the target from the provided test cases.

---

## === 3. Results

- Results are automatically saved to the `runs/` folder after each experiment.

## 📁 Directory Example

```
project_root/
├── example_symreg.py
├── runs/
│   ├── run_1/
│   ├── run_2/
│   └── ...
├── ...
```

