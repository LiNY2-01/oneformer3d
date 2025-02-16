import netron
# 
netron.start('./work_dirs/oneformer3d_1xb2_s3dis-area-5-simple-cls-nocolor/best_all_ap_50%_epoch_208.pth', address=8888)
netron.start('./work_dirs/oneformer3d_1xb2_s3dis-area-5-simple-cls-nocolor/epoch_512.pth', address=9999)
