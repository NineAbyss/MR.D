for alpha in $(seq 0 0.1 1); do
    for dataset in 'aamas_inject_03'; do
        for gae_num_layer in 2 3; do
            for gae_hidden_channels in 16 32 64; do
                echo "--alpha $alpha --gae_num_layer $gae_num_layer --gae_hidden_channels $gae_hidden_channels --real_world_name $dataset"
                python main.py --alpha $alpha --gae_num_layer $gae_num_layer --gae_hidden_channels $gae_hidden_channels --real_world_name $dataset
            done
        done
    done
done