#!/bin/bash
# usage : bash download_llama.sh 30B llama-30b 
# util 2023.3.24: 7.74MB/s speed
echo "❤️ Resume download is supported. You can ctrl-c and rerun the program to resume the downloading"
echo "you can also use bittorrent link: magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA"

PRESIGNED_URL="https://agi.gpt4.org/llama/LLaMA/*"

MODEL_SIZE=${1:-7B,13B,30B,65B}  # edit this list with the model sizes you wish to download
TARGET_FOLDER=${2:-./}           # where all files should end up

if [ $TARGET_FOLDER != "./" ]; then
    mkdir -p $TARGET_FOLDER
fi

declare -A N_SHARD_DICT

N_SHARD_DICT["7B"]="0"
N_SHARD_DICT["13B"]="1"
N_SHARD_DICT["30B"]="3"
N_SHARD_DICT["65B"]="7"

set -x
echo "Downloading tokenizer..."
wget --progress=bar:force ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
echo ✅ ${TARGET_FOLDER}"/tokenizer.model"
wget --progress=bar:force ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"
echo ✅ ${TARGET_FOLDER}"/tokenizer_checklist.chk"

(cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)

for i in ${MODEL_SIZE//,/ }
do
    echo "Downloading ${i}"
    mkdir -p ${TARGET_FOLDER}"/${i}"
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]})
    do
        #echo running: wget --continue --progress=bar:force ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
        echo "downloading file to" ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth" ...please wait for a few minutes ...
        wget --continue --progress=bar:force ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
        echo ✅ ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
    done
    wget --progress=bar:force ${PRESIGNED_URL/'*'/"${i}/params.json"} -O ${TARGET_FOLDER}"/${i}/params.json"
    echo ✅ ${TARGET_FOLDER}"/${i}/params.json"
    wget --progress=bar:force ${PRESIGNED_URL/'*'/"${i}/checklist.chk"} -O ${TARGET_FOLDER}"/${i}/checklist.chk"
    echo ✅ ${TARGET_FOLDER}"/${i}/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${i}" && md5sum -c checklist.chk)
done


