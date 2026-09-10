import os

# ============================================================
# IMPORTANT: Unsloth / PyTorch import 전에 설정해야 함
# ============================================================

# Gemma 3 12B는 RTX PRO 6000 96GB 한 장에 충분히 올라가므로,
# 평가 시 multi-GPU auto-sharding을 막아 cuda:0/cuda:1 device mismatch를 방지한다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Gemma3 Flex Attention + torch.compile 경로의 device/Triton 문제를 피한다.
os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"

# 평가 안정성을 위해 Unsloth의 torch.compile 최적화도 비활성화한다.\nos.environ["UNSLOTH_COMPILE_DISABLE"] = "1"\n
from pathlib import Path

import pandas as pd
import torch

from vlm_eval_lib_server_eager import (
    resolve_lora_path,
    prune_checkpoints_keep_last,
    load_chartqa_dataset,
    evaluate_model,
    make_summary,
    build_results_log_rows,
    save_results_log,
    build_model_config_row,
    save_model_config_log,
)

# ============================================================
# 서버 / Docker 설정
# ============================================================

# 현재 학습한 원본 모델
BASE_MODEL = "unsloth/gemma-3-12b-it"

# Unsloth Studio의 Docker 내부 기본 output 경로
LORA_SEARCH_DIR = "/root/.unsloth/studio/outputs"

# 실제 폴더:
# unsloth_gemma-3-12b-it__project-chartqa-gemma12b-l001_1788936605
#                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LORA_PROJECT_NAME = "project-chartqa-gemma12b-l001"

# 현재 실험의 파인튜닝 방식.
# QLoRA 실험에서는 "QLoRA"로만 변경.
FINE_TUNE_METHOD = "LoRA"

DATASET_NAME = "HuggingFaceM4/ChartQA"
DATASET_SPLIT = "test"

# 1차 동작 확인은 30, 최종 평가는 None 권장(전체 test 사용)
EVAL_LIMIT = 30
EVAL_SEED = 3407

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 64

# LoRA/QLoRA의 "학습 방식"만 비교하려면 추론 precision은 같아야 함.
# GPU 메모리가 충분하면 False(BF16/FP16 경로)를 권장.
# 부족하면 두 실험 모두 True로 동일하게 사용.
EVAL_LOAD_IN_4BIT = False

# 결과 저장 위치
OUTPUT_DIR = Path("/workspace/results/chartqa_gemma3_12b_lora")
RESULTS_LOG_PATH = Path("/workspace/results/unsloth_fine_tuning_results.csv")
MODEL_CONFIG_LOG_PATH = Path("/workspace/results/unsloth_fine_tuning_model_config.csv")

# ============================================================
# 체크포인트 정리
# ============================================================

# True면 checkpoint-N 중 가장 큰 N 하나만 남기고 나머지를 삭제.
KEEP_ONLY_LAST_CHECKPOINT = True

# 삭제될 목록만 보고 싶으면 True.
CHECKPOINT_CLEANUP_DRY_RUN = False

RESULTS_LOG_COLUMNS = [
    "test_name",
    "Exact Accuracy",
    "Relaxed Accuracy",
    "AVG Latency",
    "Tokens/sec",
    "Peak VRAM",
    "AVG Output Tokens",
    "AVG length",
]

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

# 완성된 adapter가 있는 최신 프로젝트 폴더를 자동 선택.
LORA_PATH, LORA_SERIAL = resolve_lora_path(
    LORA_SEARCH_DIR,
    LORA_PROJECT_NAME,
    prefer_latest_valid=True,
)


def main():
    print("Gemma 3 12B / ChartQA Base vs Fine-tuned 평가")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 감지되지 않았습니다.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("LoRA/QLoRA path:", LORA_PATH)
    print("Serial:", LORA_SERIAL)

    adapter_config_path = os.path.join(
        LORA_PATH,
        "adapter_config.json",
    )

    adapter_weight_path = os.path.join(
        LORA_PATH,
        "adapter_model.safetensors",
    )

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            "adapter_config.json을 찾을 수 없습니다:\n"
            + adapter_config_path
        )

    if not os.path.exists(adapter_weight_path):
        raise FileNotFoundError(
            "adapter_model.safetensors를 찾을 수 없습니다:\n"
            + adapter_weight_path
        )

    # --------------------------------------------------------
    # checkpoint-* 정리: 가장 마지막 checkpoint 하나만 보존
    # --------------------------------------------------------
    if KEEP_ONLY_LAST_CHECKPOINT:
        prune_checkpoints_keep_last(
            LORA_PATH,
            dry_run=CHECKPOINT_CLEANUP_DRY_RUN,
        )

    # --------------------------------------------------------
    # 결과 폴더
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CONFIG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # ChartQA TEST
    # --------------------------------------------------------
    dataset = load_chartqa_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        limit=EVAL_LIMIT,
        seed=EVAL_SEED,
    )

    print(f"평가 샘플 수: {len(dataset)}")

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------
    base_results = evaluate_model(
        model_path=BASE_MODEL,
        model_name="BASE",
        dataset=dataset,
        save_path=OUTPUT_DIR / "base_raw.csv",
        max_seq_length=MAX_SEQ_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        test_name="BASE",
        load_in_4bit=EVAL_LOAD_IN_4BIT,
    )

    # --------------------------------------------------------
    # FINE-TUNED
    # --------------------------------------------------------
    finetuned_test_name = (
        f"{FINE_TUNE_METHOD}_{LORA_PROJECT_NAME}_{LORA_SERIAL}"
    )

    finetuned_results = evaluate_model(
        model_path=LORA_PATH,
        model_name="FINE_TUNED",
        dataset=dataset,
        save_path=OUTPUT_DIR / "finetuned_raw.csv",
        max_seq_length=MAX_SEQ_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        test_name=finetuned_test_name,
        load_in_4bit=EVAL_LOAD_IN_4BIT,
    )

    # --------------------------------------------------------
    # 결과 병합
    # --------------------------------------------------------
    df = pd.DataFrame(base_results + finetuned_results)

    df.to_csv(
        OUTPUT_DIR / "all_results.csv",
        index=False,
    )

    summary = make_summary(df)

    summary.to_csv(
        OUTPUT_DIR / "summary.csv"
    )

    # --------------------------------------------------------
    # 결과 누적 로그
    # --------------------------------------------------------
    log_rows = build_results_log_rows(
        df,
        RESULTS_LOG_COLUMNS,
    )

    save_results_log(
        log_rows,
        RESULTS_LOG_PATH,
        RESULTS_LOG_COLUMNS,
    )

    # --------------------------------------------------------
    # Fine-tuned 모델 학습 설정 기록
    # --------------------------------------------------------
    model_config_row = build_model_config_row(
        lora_path=LORA_PATH,
        test_name=finetuned_test_name,
        base_model=BASE_MODEL,
        dataset_name=DATASET_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        columns=MODEL_CONFIG_COLUMNS,
        fine_tune_method=FINE_TUNE_METHOD,
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
        OUTPUT_DIR,
        "\n결과 로그 →",
        RESULTS_LOG_PATH,
        "\n모델 설정 로그 →",
        MODEL_CONFIG_LOG_PATH,
    )


if __name__ == "__main__":
    main()
