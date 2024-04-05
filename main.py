import cProfile
from trainer import Trainer
from options import TrainerOptions

options = TrainerOptions()
trainer = Trainer(options)
# cProfile.run("trainer.run()")
trainer.run()