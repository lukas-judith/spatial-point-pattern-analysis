from functional_data import *
from two_sample_testing import *
from utilities import *
    

results_dest = "Results/results_2021-08-25"

data_clca = load_file("data_clca", results_dest)
data_clcb = load_file("data_clcb", results_dest)

functions_clca = data_to_function_objects(data_clca)
functions_clcb = data_to_function_objects(data_clcb)

Function.show_functions(functions_clca+functions_clcb, folder=results_dest, save=True)

# parameters for the test
k = 10
alphas = [10, 5, 1, 0.1]


print("Starting two-sample test, approximating null distribution...")
test = SchillingTestFunctional(functions_clca+functions_clcb)
test.get_N_k_matrix(k)
test.approximate_null_distribution(10000)


for alpha in alphas:
    test.two_sample_test(alpha, folder=results_dest)
    
    
print("Done!")
