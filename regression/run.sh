datasets=('bostonHousing' 'concrete' 'energy' 'kin8nm' 'power-plant' 'protein-tertiary-structure' 'wine-quality-red' 'yacht')
model_names=('FCN' 'noisyFCN' 'fixnoiseFCN' 'dropoutFCN')
# datasets=('bostonHousing')
# model_names=('FCN')
DATE=`date +%Y-%m-%d`
mkdir "./logs/${DATE}/"

batch_size=32
optimizer_name="Adam"
learning_rate=0.005
weight_decay=0
epochs=50

for dataset_name in "${datasets[@]}";
do
    echo "dataset_name: ${dataset_name}"
    for model_name in "${model_names[@]}";
    do
        echo "model_name: ${model_name}"
        if [ "$model_name" == "noisyFCN" ]; then
            init_noiselevels=(0.01 0.05 0.1 0.2)
            for init_noiselevel in "${init_noiselevels[@]}";
            do
                echo "init_noiselevel: ${init_noiselevel}"
                log_path="./logs/${DATE}/${dataset_name}-${model_name}-${init_noiselevel}-bs${batch_size}-${optimizer_name}-lr${learning_rate}-wd${weight_decay}-epochs${epochs}.log"
                python train.py --dataset_name ${dataset_name} --model_name ${model_name} --init_noiselevel ${init_noiselevel} --batch_size ${batch_size} --optimizer_name ${optimizer_name} --learning_rate ${learning_rate} --weight_decay ${weight_decay} --epochs ${epochs} >${log_path}
            done
        elif [ "$model_name" == "fixnoiseFCN" ]; then
            fix_noiselevels=(0.01 0.05 0.1 0.2)
            for fix_noiselevel in "${fix_noiselevels[@]}";
            do
                echo "fix_noiselevel: ${fix_noiselevel}"
                log_path="./logs/${DATE}/${dataset_name}-${model_name}-${fix_noiselevel}-bs${batch_size}-${optimizer_name}-lr${learning_rate}-wd${weight_decay}-epochs${epochs}.log"
                python train.py --dataset_name ${dataset_name} --model_name ${model_name} --fix_noiselevel ${fix_noiselevel} --batch_size ${batch_size} --optimizer_name ${optimizer_name} --learning_rate ${learning_rate} --weight_decay ${weight_decay} --epochs ${epochs} >${log_path}
            done
        elif [ "$model_name" == "dropoutFCN" ]; then
            dropout_probs=(0.005 0.05)
            for dropout_prob in "${dropout_probs[@]}";
            do
                echo "dropout_prob: ${dropout_prob}"
                log_path="./logs/${DATE}/${dataset_name}-${model_name}-${dropout_prob}-bs${batch_size}-${optimizer_name}-lr${learning_rate}-wd${weight_decay}-epochs${epochs}.log"
                python train.py --dataset_name ${dataset_name} --model_name ${model_name} --dropout_prob ${dropout_prob} --batch_size ${batch_size} --optimizer_name ${optimizer_name} --learning_rate ${learning_rate} --weight_decay ${weight_decay} --epochs ${epochs} >${log_path}
            done
        else
            echo "FCN"
            log_path="./logs/${DATE}/${dataset_name}-${model_name}-bs${batch_size}-${optimizer_name}-lr${learning_rate}-wd${weight_decay}-epochs${epochs}.log"
            python train.py --dataset_name ${dataset_name} --model_name ${model_name} --batch_size ${batch_size} --optimizer_name ${optimizer_name} --learning_rate ${learning_rate} --weight_decay ${weight_decay} --epochs ${epochs} >${log_path}
        fi
    done
done