# python archive_non_rtde_main/tools/visualize_zed_target_debug.py \
#   --config configs/handover_zed_single.yaml \
#   --show-2d --show-depth --show-seg-input


# # 투명 object의 픽셀 대비를 전처리 전과 후 비교
# python archive_non_rtde_main/tools/visualize_zed_target_debug.py \
#   --config configs/handover_zed_single.yaml \
#   --show-2d --show-seg-input --show-contrast-debug --show-depth

# 실제 전처리 전 이미지와 전처리 후 이미지를 따로 segmentation해서 비교
python archive_non_rtde_main/tools/visualize_zed_target_debug.py \
  --config configs/handover_zed_single.yaml \
  --show-2d --show-seg-input --show-contrast-debug --show-seg-compare --show-depth
