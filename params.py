
NUM_EPOCHS = 20_000
PATIENCE = 2000
net_choices = [
    dict(learning_rate=0.001, num_epochs=NUM_EPOCHS, patience_epochs=PATIENCE,
         model_type="STONet", embedding_dim=100,
         branch_actf = 'Tanh', branch_width=100, num_branch_layers=8,
         trunk_actf = 'Tanh', trunk_width=100, num_trunk_layers=8,
         root_actf = 'Tanh', root_width=100, num_root_layers=1,
         stonet_attention_actf = 'Tanh', num_stonet_attention_blocks=4,
         fourier=False),
    # TODO: you can add more models for comparison
    # dict(learning_rate=0.001, num_epochs=NUM_EPOCHS, patience_epochs=PATIENCE,
    #      model_type="STONet", embedding_dim=50,
    #      branch_actf = 'Tanh', branch_width=50, num_branch_layers=8,
    #      trunk_actf = 'Tanh', trunk_width=50, num_trunk_layers=8,
    #      root_actf = 'Tanh', root_width=50, num_root_layers=1,
    #      stonet_attention_actf = 'Tanh', num_stonet_attention_blocks=4,
    #      fourier=False),
]
