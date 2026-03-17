set -a
source ~/.egoverse_env
set +a

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="auto"
export AWS_REGION="auto"

# aria download id
ID=1766002714075
path="/home/ubuntu/aria_proc_download"


# s5cmd --endpoint-url "$R2_ENDPOINT_URL" sync \
#   "s3://rldb/raw_v2/test_aria/mps_${ID}_vrs/**" \
#   "${path}/mps_${ID}_vrs/"

# s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/test_aria/${ID}.json" $path
# s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/test_aria/${ID}.vrs" $path
# s5cmd --endpoint-url "uR2_ENDPOINT_URL" cp "s3://rldb/raw_v2/test_aria/${ID}_metadata.json" $path

# delete processed directories
# s5cmd --endpoint-url "$R2_ENDPOINT_URL" rm "s3://rldb/processed_v3/test_eva/**"

# download processed zarrs and mp4s
# s5cmd --endpoint-url "$R2_ENDPOINT_URL" sync \
#   "s3://rldb/processed_v3/test_aria/**" \
#   "$path"

eva_ID=1767406562519
# # eva move hdf5
# # s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/eva/${eva_ID}.hdf5" "s3://rldb/raw_v2/test_eva2/${eva_ID}.hdf5"

# s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp \
#     "s3://rldb/raw_v2/eva/${eva_ID}.hdf5" \
#     "/home/ubuntu/up_res_download/${eva_ID}.hdf5"

# # Then upload to destination
# s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp \
#     "/home/ubuntu/up_res_download/${eva_ID}.hdf5" \
#     "s3://rldb/raw_v2/test_eva/${eva_ID}.hdf5"

# # Clean up
# rm /home/ubuntu/up_res_download/${eva_ID}.hdf5