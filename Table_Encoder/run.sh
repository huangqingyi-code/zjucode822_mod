CUDA_VISIBLE_DEVICES=0 nohup python train.py --pred_type generation --log_activate > logs/generation.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --pred_type classification --log_activate > logs/classification.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --pred_type contrastive --data_path /data0/datasets/spider/database_csv_filtered_by_col_name/semantic --from_csv --batch_size 32
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --pred_type contrastive --data_path /home/llong/Table-Encoder/data/data_list.npz --from_csv --batch_size 32

nohup python train.py --pred_type contrastive --log_activate True > logs/contrastive_1.log 2>&1 &

nohup python train.py --pred_type contrastive --log_activate True --data_path /data0/datasets/embedded_tables_spider --load_model True --model_path checkpoints_save/encoder-contrastive-100d-50epochs-0.0001lr.pt > logs/contrastive_spider.log 2>&1 &

nohup python train.py --pred_type contrastive --log_activate True --data_path /data0/datasets/embedded_tables_spider > logs/contrastive_spider.log 2>&1 &

python train.py --pred_type contrastive --data_path /data0/datasets/embedded_tables_spider --load_model True --model_path checkpoints_save/encoder-contrastive-100d-50epochs-0.0001lr.pt

python train.py --pred_type contrastive --data_path /data0/datasets/embedded_tables_spider

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py --pred_type contrastive --data_path /mnt/data/sxj/datasets/spider/database_csv_flat --log_activate --from_csv --batch_size 8 > logs/contrastive_spider_joint.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py --pred_type contrastive --data_path /data0/datasets/spider/database_csv_filtered_by_col_name/semantic --log_activate --from_csv --batch_size 32 > logs/contrastive_spider_new.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py --pred_type contrastive --data_path /home/llong/Table-Encoder/data/data_list.npz --log_activate --from_csv --batch_size 32 > logs/contrastive_full_new.log 2>&1 &

python train.py --pred_type contrastive --data_path /home/llong/Table-Encoder/data/data_list.npz --from_csv --batch_size 32

CUDA_VISIBLE_DEVICES=3,5 python train.py --data_path /home/llong/Table-Encoder/data --from_csv --comment test
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py --data_path /home/llong/Table-Encoder/data --from_csv --log_activate --comment bge > logs/0720_bge.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py --pred_type contrastive --data_path /home/llong/Table-Encoder/data/full_data.npz --log_activate --from_csv --batch_size 2 > logs/contrastive_full_0706.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --pred_type contrastive --data_path /home/llong/Table-Encoder/data/spider_only.npz --from_csv --batch_size 10 --log_interval 1 --freeze_st