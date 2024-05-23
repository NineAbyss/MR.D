for alpha in 0.2; do
    for dataset in 'aamas_inject_03'; do
        for gae_num_layer in 3; do
            for gae_hidden_channels in 64; do
                for gae_epochs in 2000; do
                    for gcl_hidden_dim in 16 32 64; do  
                        for gcl_num_layer in 2 3; do  
                            echo " --alpha $alpha --gae_num_layer $gae_num_layer --gae_hidden_channels $gae_hidden_channels --real_world_name $dataset --gae_epochs $gae_epochs --gcl_hidden_dim $gcl_hidden_dim --gcl_num_layer $gcl_num_layer"
                            python main.py --alpha $alpha --gae_num_layer $gae_num_layer --gae_hidden_channels $gae_hidden_channels --real_world_name $dataset --ours 1 --gae_epochs $gae_epochs --gcl_hidden_dim $gcl_hidden_dim --gcl_num_layer $gcl_num_layer
                        done
                    done
                done
            done
        done
    done
done