import cPickle as pickle
with open('/var/www/songchuan/comment-generate/prototype3.2/neural-editor-data/edit_runs/5/checkpoints/83000.checkpoint/metadata.p') as f:
	d = pickle.load(f)
print d