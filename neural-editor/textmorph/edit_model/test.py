import argparse
from gtd.utils import Config
from textmorph.edit_model.training_run import EditTrainingRuns

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('exp_id', nargs='+')
arg_parser.add_argument('-c', '--check_commit', default='strict')
args = arg_parser.parse_args()

experiments = EditTrainingRuns(check_commit=(args.check_commit=='strict'))
print 'test0'

exp_id = args.exp_id
if exp_id == ['default']:
    # new default experiment
    exp = experiments.new()
elif len(exp_id) == 1 and exp_id[0].isdigit():
    # reload old experiment
    exp = experiments[int(exp_id[0])]
else:
    # new experiment according to configs
    config = Config.from_file(exp_id[0])
    print 'test1'
    for filename in exp_id[1:]:
        config = Config.merge(config, Config.from_file(filename))
    print 'test2'
    exp = experiments.new(config)  # new experiment from config
    print 'test'

print exp
print type(exp)