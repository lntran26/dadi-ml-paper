import dadi
import pickle

# input_fname = 'data/test_100_theta_1000_ns39'
# output_fname = 'data/projected_test_100_theta_1000_ns39'
# projected_size = (20,20)

input_fname = 'data/test_100_theta_100_ns39'
output_fname = 'data/projected_test_100_theta_100_ns39'
projected_size = (20,20)

test_dict = pickle.load(open(input_fname, 'rb'))
projected_test_dict = {}

for params_key in test_dict:
    fs = test_dict[params_key]
    projected_fs = fs.project(projected_size)
    projected_test_dict[params_key] = projected_fs
    
pickle.dump(projected_test_dict, open(output_fname, 'wb'))