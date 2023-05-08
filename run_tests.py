


## Helper Functions to run the tests
def test_baseline(task_name, method, task_params):
    '''task_name: str in {'so2', 'smd', 'so3', 'motion', 'accel'}
    method: str in {'zoh', 'ls_opt', 'kf', 'kf_ss', 'kf_smooth', 'id_kf', 'id_kf_sim', 'learn_kf'}
    task_params: dict {param:value} that are passed to generate trajectories
    '''
    pass

def train_test_model(task_name, method, model_params):
    pass