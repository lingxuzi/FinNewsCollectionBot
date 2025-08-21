import swanlab

class LogAgent:
    def __init__(self, project_name, config, name='vision_price_trend'):
        swanlab.login(api_key='U8RYQtw9BGcj2r4yt6cD1')

        swanlab.init(project=project_name, config=config, experiment_name=name, reinit=True, id=f'{project_name}{name}', resume='allow')

    def log(self, info):
        swanlab.log(info)