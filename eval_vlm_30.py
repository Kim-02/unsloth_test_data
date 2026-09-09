import os
from pathlib import Path

import pandas as pd
import torch

from vlm_eval_lib import (
    resolve_lora_path,
    load_prepared_dataset,
    evaluate_model,
    add_accuracy_metrics,
    make_summary,
    build_results_log_rows,
    save_results_log,
    build_model_config_row,
    save_model_config_log,
)


# ============================================================
# 사용자 설정
# ============================================================

BASE_MODEL = (
    "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
)

# LoRA 폴더들이 모여 있는 상위 디렉토리
LORA_SEARCH_DIR = "/home/vic06/Desktop/unsloth_models"

# 폴더명 중간에 들어가는 프로젝트 식별자
# (예: ..._4bit__project-qwen3.5vl-2b-construction-lora_1788759647
#                -------- 이 부분 ----------------)
LORA_PROJECT_NAME = "project-qwen3.5vl-2b-construction-lora"

DATASET_NAME = "LouisChen15/ConstructionSite"

# 이미 준비된 테스트셋(selected_30.csv + images/)이 들어있는 폴더
OUTPUT_DIR = Path("construction_vlm_eval30")

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 192

# 채팅에서 사용했던 것과 동일한 Prompt
PROMPT = (
    "Describe this construction site in 3 to 5 sentences."
    "Focus on workers, personal protective equipment, machinery,"
    "construction materials, and visible activities."
    "Only describe what is clearly visible in the image."
)

# unsloth_fine_tuning_data.ods의 "결과" 시트와 동일한 형식으로
# 누적 기록되는 로그 파일
RESULTS_LOG_PATH = Path("unsloth_fine_tuning_results.csv")

RESULTS_LOG_COLUMNS = [
    "test_name",
    "BERTScore F1",
    "ROUGE-L F1",
    "Domain Precision",
    "Domain Recall",
    "Domain F1",
    "AVG Latency",
    "Tokens/sec",
    "Peak VRAM",
    "AVG Output Tokens",
    "AVG length",
    "Domain terms",
    "Lexical diversity",
]

# unsloth_fine_tuning_data.ods의 "모델 설정" 시트와 동일한 형식으로
# 누적 기록되는 로그 파일 (Fine-tuned 모델의 학습 설정)
MODEL_CONFIG_LOG_PATH = Path(
    "unsloth_fine_tuning_model_config.csv"
)

MODEL_CONFIG_COLUMNS = [
    "test_name",
    "모델",
    "방법",
    "데이터셋",
    "스탭/에포크",
    "컨텍스트 길이",
    "학습률",
    "랭크",
    "알파",
    "드롭아웃",
    "비전 레이어",
    "언어 레이어",
    "어텐션 모듈",
    "MLP모듈",
    "LoRA종류",
    "옵티마이저",
    "LR 스케줄러",
    "배치 크기",
    "그래디언트 누적",
    "가중치 감쇠",
    "워밍업 스텝",
    "에포크",
    "시드",
    "이미지 사이즈",
    "그래디언트 체크포인팅",
    "어시스턴트 응답만 학습",
]


# 추론에 사용할 LoRA 경로 / 일련번호
LORA_PATH, LORA_SERIAL = resolve_lora_path(
    LORA_SEARCH_DIR,
    LORA_PROJECT_NAME,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Qwen3-VL Base vs Fine-tuned 평가")

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU가 감지되지 않았습니다."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    # LoRA 경로 확인
    if not os.path.exists(LORA_PATH):

        raise FileNotFoundError(
            f"LoRA 경로가 없습니다: {LORA_PATH}"
        )

    adapter_config_path = os.path.join(
        LORA_PATH,
        "adapter_config.json",
    )

    if not os.path.exists(adapter_config_path):

        raise FileNotFoundError(
            "adapter_config.json을 찾을 수 없습니다:\n"
            + adapter_config_path
        )

    print("LoRA Adapter:", adapter_config_path)

    # ========================================================
    # DATASET (이미 준비된 테스트셋을 그대로 로드)
    # ========================================================

    dataset = load_prepared_dataset(OUTPUT_DIR)

    # ========================================================
    # BASE
    # ========================================================

    base_results = evaluate_model(
        BASE_MODEL,
        "BASE",
        dataset,
        OUTPUT_DIR / "base_raw.csv",
        MAX_SEQ_LENGTH,
        PROMPT,
        MAX_NEW_TOKENS,
        LORA_PROJECT_NAME,
    )

    # ========================================================
    # FINE-TUNED
    # ========================================================

    finetuned_results = evaluate_model(
        LORA_PATH,
        "FINE_TUNED",
        dataset,
        OUTPUT_DIR / "finetuned_raw.csv",
        MAX_SEQ_LENGTH,
        PROMPT,
        MAX_NEW_TOKENS,
        LORA_PROJECT_NAME,
    )

    # ========================================================
    # 결과 병합 + 정확도 계산
    # ========================================================

    df = pd.DataFrame(
        base_results
        + finetuned_results
    )

    df = add_accuracy_metrics(df)

    df.to_csv(
        OUTPUT_DIR
        / "all_results.csv",
        index=False,
    )

    # ========================================================
    # 평균 결과
    # ========================================================

    summary = make_summary(df)

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv"
    )

    # ========================================================
    # unsloth_fine_tuning_data.ods 형식으로 결과 누적 기록
    # ========================================================

    log_rows = build_results_log_rows(
        df,
        RESULTS_LOG_COLUMNS,
    )

    save_results_log(
        log_rows,
        RESULTS_LOG_PATH,
        RESULTS_LOG_COLUMNS,
    )

    # ========================================================
    # unsloth_fine_tuning_data.ods "모델 설정" 형식으로 기록
    # ========================================================

    model_config_row = build_model_config_row(
        LORA_PATH,
        LORA_PROJECT_NAME,
        BASE_MODEL,
        DATASET_NAME,
        MAX_SEQ_LENGTH,
        MODEL_CONFIG_COLUMNS,
    )

    save_model_config_log(
        model_config_row,
        MODEL_CONFIG_LOG_PATH,
        MODEL_CONFIG_COLUMNS,
    )

    print("\n최종 결과")
    print(summary)

    print(
        "\n완료 →",
        OUTPUT_DIR.resolve(),
        f"/ {RESULTS_LOG_PATH} / {MODEL_CONFIG_LOG_PATH}",
    )


if __name__ == "__main__":
    main()
