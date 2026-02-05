import wandb

api = wandb.Api()
# Replace , <project>, and <id> with your specific run details
run = api.run('single-agent-binary')
file = run.file('output.log')
file.download()
