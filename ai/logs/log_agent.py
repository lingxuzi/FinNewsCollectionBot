import swanlab

class LogAgent:
    def __init__(self, project_name, config):
        swanlab.login(api_key='U8RYQtw9BGcj2r4yt6cD1')

        swanlab.init(project=project_name, config=config)

    def log(self, info):
        swanlab.log(info)