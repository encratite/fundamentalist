import cProfile
from trainer import Trainer
from options import TrainerOptions

options = TrainerOptions()
trainer = Trainer(options)
if Trainer.USE_CPROFILE:
	cProfile.run("trainer.run()")
else:
	trainer.run()