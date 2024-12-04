#!/bin/bash

# Google Drive 파일 ID
TEST_TRANSACTION_FILE_ID="1MpMFeGNvODOEx34aMWecaxzhZd8JEp8j"
TRAIN_TRANSACTION_FILE_ID="1tD3IZWXZxOGrvTmWRYPbIncRW9dSZ49C"
TEST_IDENTITY_FILE_ID="1JtfpNoB0u8N80UIcnGPBVyIMdaks_j3D"
TRAIN_IDENTITY_FILE_ID="1M4_kvduTfuhRUxVpDJJnxcRYPrjFVvmO"

# 다운로드할 파일 이름
TEST_TRANSACTION_FILE_NAME="data/test_transaction.csv"
TRAIN_TRANSACTION_FILE_NAME="data/train_transaction.csv"
TEST_IDENTITY_FILE_NAME="data/test_identity.csv"
TRAIN_IDENTITY_FILE_NAME="data/train_identity.csv"

# Google Drive에서 파일 다운로드
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${TEST_TRANSACTION_FILE_ID}" -O ${TEST_TRANSACTION_FILE_NAME}
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${TRAIN_TRANSACTION_FILE_ID}" -O ${TRAIN_TRANSACTION_FILE_NAME}
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${TEST_IDENTITY_FILE_ID}" -O ${TEST_IDENTITY_FILE_NAME}
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${TRAIN_IDENTITY_FILE_ID}" -O ${TRAIN_IDENTITY_FILE_NAME}