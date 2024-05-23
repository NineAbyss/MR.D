for alpha in $(seq 0 0.1 1); do
    for dataset in 'wu_inject_03' 'aamas_inject_03'; do
        for gae_num_layer in 2 3; do
            for gae_hidden_channels in 16 32 64; do
                for gae_epochs in 2000 5000; do
                    echo " --alpha $alpha --gae_num_layer $gae_num_layer --gae_hidden_channels $gae_hidden_channels --real_world_name $dataset --gae_epochs $gae_epochs"
                    python main.py --alpha $alpha --gae_num_layer $gae_num_layer --gae_hidden_channels $gae_hidden_channels --real_world_name $dataset --ours 1 --gae_epochs $gae_epochs
                done
            done
        done
    done
done