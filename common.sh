## g_o
python train_gs.py -s data/R63_two_view -m output/gs_init/my_car -r 4 --sparse_view_num 2 --sh_degree 2 --init_pcd_name Point3D --white_background --random_background

python render.py \
    -m output/gs_init/my_car \
    --sparse_view_num 2 --sh_degree 2 \
    --init_pcd_name Point3D \
    --white_background --skip_all --skip_train

python render.py \
    -m output/gs_init/my_car \
    --sparse_view_num 2 --sh_degree 2 \
    --init_pcd_name Point3D \
    --white_background --render_path