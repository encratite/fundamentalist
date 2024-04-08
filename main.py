import cProfile
import multiprocessing
from trainer import Trainer
from options import TrainerOptions

def run():
	options = TrainerOptions()
	trainer = Trainer(options)
	if Trainer.USE_CPROFILE:
		cProfile.run("trainer.run()")
	else:
		trainer.run()

if __name__ == '__main__':
	multiprocessing.freeze_support()
	run()