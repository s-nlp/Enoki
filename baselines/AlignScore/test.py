from AlignScore.src.alignscore import AlignScore

scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda', ckpt_path='./checkpoints/AlignScore-large.ckpt', evaluation_mode='nli_sp')
score = scorer.score(contexts=['hello world.'], claims=['hello world.'])

print(score)


scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda', ckpt_path='./checkpoints/AlignScore-large.ckpt', evaluation_mode='bin_sp')
score = scorer.score(contexts=['Mike is an elderly person', 'Mike is an elderly person'], claims=['Mike is old', 'Mike is an young'])

print(score)