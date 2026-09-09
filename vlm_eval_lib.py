import os
import re
import gc
import json
import time

import torch
import pandas as pd
import numpy as np
from PIL import Image

from unsloth import FastVisionModel
from rouge_score import rouge_scorer


# ============================================================
# LoRA 폴더 탐색
# ============================================================

def resolve_lora_path(search_dir, project_name):
    """
    search_dir 안에서 폴더명에
    '{project_name}_<일련번호>'가 포함된 폴더를 찾는다.

    - 0개  : FileNotFoundError
    - 2개+ : RuntimeError (이름 + 일련번호 목록을 보여줌)
    - 1개  : 해당 폴더 경로 반환
    """

    if not os.path.isdir(search_dir):
        raise FileNotFoundError(
            f"LoRA 검색 폴더가 없습니다: {search_dir}"
        )

    pattern = re.compile(
        re.escape(project_name) + r"_(\d+)$"
    )

    candidates = []

    for name in sorted(os.listdir(search_dir)):

        full_path = os.path.join(search_dir, name)

        if not os.path.isdir(full_path):
            continue

        match = pattern.search(name)

        if match:
            candidates.append(
                (name, match.group(1), full_path)
            )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"'{project_name}' 이름을 가진 LoRA 폴더를 "
            f"{search_dir} 에서 찾지 못했습니다."
        )

    if len(candidates) > 1:
        listing = "\n".join(
            f"  - {name}  (일련번호: {serial})"
            for name, serial, _ in candidates
        )
        raise RuntimeError(
            f"'{project_name}' 프로젝트의 LoRA 폴더가 "
            f"{len(candidates)}개 발견되었습니다. 하나만 남기거나 "
            f"일련번호를 지정해서 다시 실행하세요.\n{listing}"
        )

    return candidates[0][2], candidates[0][1]


# ============================================================
# 건설 도메인 표현 사전
# ============================================================

DOMAIN_TERMS = {
    "worker": [
        "worker",
        "workers",
        "construction worker",
        "construction workers",
        "laborer",
        "laborers",
    ],

    "hard_hat": [
        "hard hat",
        "hard hats",
        "helmet",
        "helmets",
        "safety helmet",
        "safety helmets",
    ],

    "safety_vest": [
        "safety vest",
        "safety vests",
        "reflective vest",
        "reflective vests",
        "high visibility vest",
        "high-visibility vest",
    ],

    "rebar": [
        "rebar",
        "rebar grid",
        "reinforcing bar",
        "reinforcing bars",
        "reinforcement bar",
        "reinforcement bars",
        "steel reinforcement",
    ],

    "excavator": [
        "excavator",
        "excavators",
    ],

    "crane": [
        "crane",
        "cranes",
    ],

    "concrete_pump": [
        "concrete pump",
        "concrete pumps",
        "concrete pump truck",
    ],

    "scaffolding": [
        "scaffold",
        "scaffolds",
        "scaffolding",
    ],

    "formwork": [
        "formwork",
        "concrete form",
        "concrete forms",
    ],

    "concrete": [
        "concrete",
    ],

    "ladder": [
        "ladder",
        "ladders",
    ],

    "steel": [
        "steel",
    ],

    "wall": [
        "wall",
        "walls",
    ],

    "reinforcement": [
        "reinforcement",
        "reinforced",
    ],
}


# ============================================================
# GPU 메모리 정리 / 이미지 전처리
# ============================================================

def clear_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resize_for_vlm(image, max_side=1024):
    """
    너무 큰 이미지는 비율을 유지하면서 최대 변을 1024px로 축소.
    작은 이미지는 확대하지 않음.
    """
    image = image.convert("RGB")

    width, height = image.size

    if max(width, height) <= max_side:
        return image

    scale = max_side / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize(
        (new_width, new_height)
    )


# ============================================================
# Domain Term 추출
# ============================================================

def extract_domain_terms(text):
    text = str(text).lower()

    found = set()

    for category, aliases in DOMAIN_TERMS.items():

        for alias in aliases:

            pattern = (
                r"\b"
                + re.escape(alias.lower())
                + r"\b"
            )

            if re.search(pattern, text):
                found.add(category)
                break

    return found


# ============================================================
# Domain Precision / Recall / F1
# ============================================================

def domain_term_metrics(gt, pred):

    gt_terms = extract_domain_terms(gt)
    pred_terms = extract_domain_terms(pred)

    if len(gt_terms) == 0:
        return np.nan, np.nan, np.nan

    tp = len(gt_terms & pred_terms)
    fp = len(pred_terms - gt_terms)
    fn = len(gt_terms - pred_terms)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


# ============================================================
# 표현도 측정
# ============================================================

def lexical_metrics(text):

    words = re.findall(
        r"[A-Za-z0-9'-]+",
        str(text).lower(),
    )

    word_count = len(words)

    if word_count == 0:
        return 0, 0, 0.0

    unique_words = len(set(words))

    lexical_diversity = (
        unique_words / word_count
    )

    return (
        word_count,
        unique_words,
        lexical_diversity,
    )


# ============================================================
# 준비된 테스트셋 로드
# ============================================================

def load_prepared_dataset(output_dir):
    """
    output_dir/selected_30.csv 와 output_dir/images/*.jpg를
    그대로 불러온다 (HF 데이터셋 재다운로드 없음).
    """

    meta = pd.read_csv(
        os.path.join(output_dir, "selected_30.csv")
    )

    samples = []

    for _, row in meta.iterrows():

        image = Image.open(row["image"])

        samples.append({
            "image": image,
            "image_id": row.get("image_id", ""),
            "image_caption": row.get("gt_caption", ""),
        })

    return samples


# ============================================================
# 모델 로드
# ============================================================

def load_model(model_path, model_name, max_seq_length):

    print(f"\n[{model_name}] 모델 로딩: {model_path}")

    clear_gpu()

    model, processor = (
        FastVisionModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
    )

    # 추론 모드
    FastVisionModel.for_inference(model)

    return model, processor


# ============================================================
# VLM 입력 생성
# ============================================================

def make_inputs(processor, image, prompt):

    # ------------------------------------------------
    # 고해상도 이미지 visual token 폭증 방지
    # ------------------------------------------------
    image = resize_for_vlm(
        image,
        max_side=1024,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=input_text,

        # 중요:
        # image token 중간이 잘리면 안 됨
        truncation=False,

        add_special_tokens=False,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    return inputs


# ============================================================
# 이미지 1장 추론
# ============================================================

def inference_one(
    model,
    processor,
    image,
    prompt,
    max_new_tokens,
):

    inputs = make_inputs(
        processor,
        image,
        prompt,
    )

    input_length = (
        inputs["input_ids"].shape[1]
    )
    input_tokens = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        output = model.generate(
            **inputs,

            max_new_tokens=max_new_tokens,

            # Base/Fine 모델 비교이므로
            # 랜덤 sampling 제거
            do_sample=False,

            use_cache=True,
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    generated = output[
        :,
        input_length:
    ]

    output_tokens = (
        generated.shape[1]
    )

    prediction = (
        processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]
        .strip()
    )

    tokens_per_sec = (
        output_tokens / elapsed
        if elapsed > 0
        else 0
    )

    peak_vram_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    del inputs
    del output
    del generated

    return {
        "prediction": prediction,
        "latency_sec": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_gb": peak_vram_gb,
    }


# ============================================================
# Warm-up
# ============================================================

def warmup(model, processor, dataset, prompt):

    image = resize_for_vlm(
        dataset[0]["image"],
        max_side=1024,
    )

    inputs = make_inputs(
        processor,
        image,
        prompt,
    )

    with torch.inference_mode():

        _ = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()

    del inputs

    clear_gpu()


# ============================================================
# test_name 결정
# ============================================================

def get_test_name(model_name, lora_project_name):
    """
    FINE_TUNED → 추론에 사용한 LoRA 가중치 폴더의 가운데 이름
    BASE       → LoRA를 쓰지 않으므로 "BASE" 고정
    """
    return (
        lora_project_name
        if model_name == "FINE_TUNED"
        else "BASE"
    )


# ============================================================
# 모델 하나에 대해 데이터셋 전체 추론
# ============================================================

def evaluate_model(
    model_path,
    model_name,
    dataset,
    save_path,
    max_seq_length,
    prompt,
    max_new_tokens,
    lora_project_name,
):

    test_name = get_test_name(model_name, lora_project_name)

    num_samples = len(dataset)

    model, processor = load_model(
        model_path,
        model_name,
        max_seq_length,
    )

    warmup(
        model,
        processor,
        dataset,
        prompt,
    )

    results = []

    print(f"[{model_name}] {num_samples}장 추론 시작")

    for i, sample in enumerate(dataset):

        image = resize_for_vlm(
            sample["image"],
            max_side=1024,
        )

        gt_caption = sample.get(
            "image_caption",
            "",
        )

        result = inference_one(
            model,
            processor,
            image,
            prompt,
            max_new_tokens,
        )

        # ---------------------------------
        # 표현도
        # ---------------------------------

        (
            word_count,
            unique_words,
            lexical_diversity,
        ) = lexical_metrics(
            result["prediction"]
        )

        domain_terms = extract_domain_terms(
            result["prediction"]
        )

        # ---------------------------------
        # Domain 정확도
        # ---------------------------------

        (
            domain_precision,
            domain_recall,
            domain_f1,
        ) = domain_term_metrics(
            gt_caption,
            result["prediction"],
        )

        row = {
            "sample_index": i,

            "image_id": sample.get(
                "image_id",
                "",
            ),

            # BASE / FINE_TUNED 구분은 test_name 하나로 처리
            "test_name": test_name,

            "gt_caption": gt_caption,

            "prediction":
                result["prediction"],

            # 정확도
            "domain_precision":
                domain_precision,

            "domain_recall":
                domain_recall,

            "domain_f1":
                domain_f1,

            # 연산량
            "latency_sec":
                result["latency_sec"],

            "output_tokens":
                result["output_tokens"],

            "tokens_per_sec":
                result["tokens_per_sec"],

            "peak_vram_gb":
                result["peak_vram_gb"],

            # 표현도
            "word_count":
                word_count,

            "unique_word_count":
                unique_words,

            "lexical_diversity":
                lexical_diversity,

            "domain_term_count":
                len(domain_terms),

            "domain_terms":
                ", ".join(
                    sorted(domain_terms)
                ),
        }

        results.append(row)
        pd.DataFrame(results).to_csv(
            save_path,
            index=False,
        )
        print(
            f"[{model_name}] "
            f"{i+1:02d}/{num_samples} | "
            f"input={result['input_tokens']} tok | "
            f"{result['latency_sec']:.2f} sec | "
            f"{result['tokens_per_sec']:.2f} tok/s | "
            f"{result['peak_vram_gb']:.2f} GB"
        )

    # GPU에서 모델 제거
    del model
    del processor

    clear_gpu()

    return results


# ============================================================
# ROUGE-L / BERTScore
# ============================================================

def add_accuracy_metrics(df):

    # -----------------------------
    # ROUGE-L
    # -----------------------------

    print("ROUGE-L 계산 중...")

    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    rouge_values = []

    for _, row in df.iterrows():

        score = rouge.score(
            str(row["gt_caption"]),
            str(row["prediction"]),
        )

        rouge_values.append(
            score["rougeL"].fmeasure
        )

    df["rougeL_f1"] = rouge_values

    # -----------------------------
    # BERTScore
    # -----------------------------

    print("BERTScore 계산 중...")

    # VLM은 이미 GPU에서 제거된 상태
    clear_gpu()

    from bert_score import score

    candidates = (
        df["prediction"]
        .astype(str)
        .tolist()
    )

    references = (
        df["gt_caption"]
        .astype(str)
        .tolist()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    P, R, F1 = score(
        candidates,
        references,
        lang="en",
        device=device,
        verbose=True,
    )

    df["bertscore_precision"] = (
        P.cpu().numpy()
    )

    df["bertscore_recall"] = (
        R.cpu().numpy()
    )

    df["bertscore_f1"] = (
        F1.cpu().numpy()
    )

    clear_gpu()

    return df


# ============================================================
# 최종 평균
# ============================================================

def make_summary(df):

    metrics = [

        # 정확도
        "bertscore_f1",
        "rougeL_f1",
        "domain_precision",
        "domain_recall",
        "domain_f1",

        # 연산량
        "latency_sec",
        "tokens_per_sec",
        "peak_vram_gb",
        "output_tokens",

        # 표현도
        "domain_term_count",
        "word_count",
        "unique_word_count",
        "lexical_diversity",
    ]

    summary = (
        df
        .groupby("test_name")[metrics]
        .mean(numeric_only=True)
        .round(4)
    )

    return summary


# ============================================================
# unsloth_fine_tuning_data.ods "결과" 시트 형식으로 변환
# ============================================================

def build_results_log_rows(df, columns):
    """
    all_results 형태의 df(test_name별 상세 결과)를 받아서
    unsloth_fine_tuning_data.ods의 "결과" 시트와 동일한
    컬럼 형식으로 test_name별 1행씩 만든다.
    """

    rows = []

    for test_name, group in df.groupby("test_name"):

        rows.append({
            "test_name":
                test_name,

            "BERTScore F1":
                round(group["bertscore_f1"].mean(), 4),

            "ROUGE-L F1":
                round(group["rougeL_f1"].mean(), 4),

            "Domain Precision":
                round(group["domain_precision"].mean(), 4),

            "Domain Recall":
                round(group["domain_recall"].mean(), 4),

            "Domain F1":
                round(group["domain_f1"].mean(), 4),

            "AVG Latency":
                f"{group['latency_sec'].mean():.2f} s",

            "Tokens/sec":
                round(group["tokens_per_sec"].mean(), 2),

            "Peak VRAM":
                f"{group['peak_vram_gb'].mean():.2f} GB",

            "AVG Output Tokens":
                round(group["output_tokens"].mean(), 1),

            "AVG length":
                round(group["word_count"].mean(), 1),

            "Domain terms":
                round(group["domain_term_count"].mean(), 2),

            "Lexical diversity":
                round(group["lexical_diversity"].mean(), 4),
        })

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def save_results_log(new_rows, path, columns):
    """
    path에 결과를 누적 기록한다.
    같은 test_name이 이미 있으면 그 행을 새 결과로 덮어쓰고,
    없으면 새 행으로 추가한다.
    """

    if os.path.exists(path):

        existing = pd.read_csv(path)

        existing = existing[
            ~existing["test_name"].isin(
                new_rows["test_name"]
            )
        ]

        combined = pd.concat(
            [existing, new_rows],
            ignore_index=True,
        )

    else:
        combined = new_rows

    combined = combined[columns]

    combined.to_csv(path, index=False)

    return combined


# ============================================================
# unsloth_fine_tuning_data.ods "모델 설정" 시트 형식으로 변환
# ============================================================

def load_lora_training_config(lora_path):
    """
    LoRA 폴더의 adapter_config.json / training_args.bin을 읽는다.
    training_args.bin이 없거나 읽기 실패하면 빈 dict를 반환한다.
    """

    with open(
        os.path.join(lora_path, "adapter_config.json")
    ) as f:
        adapter_config = json.load(f)

    training_args = {}

    training_args_path = os.path.join(
        lora_path, "training_args.bin"
    )

    if os.path.exists(training_args_path):

        try:
            args_obj = torch.load(
                training_args_path,
                weights_only=False,
            )
            training_args = args_obj.to_dict()

        except Exception as e:
            print(
                f"training_args.bin 읽기 실패 "
                f"(모델 설정 일부 항목은 빈칸으로 남음): {e}"
            )

    return adapter_config, training_args


def format_optimizer_name(optim):

    if not optim:
        return ""

    mapping = {
        "adamw_8bit": "AdamW 8-bit",
        "adamw_bnb_8bit": "AdamW 8-bit",
        "paged_adamw_8bit": "Paged AdamW 8-bit",
        "paged_adamw_32bit": "Paged AdamW 32-bit",
        "adamw_torch": "AdamW",
        "adamw_torch_fused": "AdamW",
        "sgd": "SGD",
        "adafactor": "Adafactor",
    }

    return mapping.get(optim, optim)


def build_model_config_row(
    lora_path,
    test_name,
    base_model,
    dataset_name,
    max_seq_length,
    columns,
):
    """
    lora_path 기준 unsloth_fine_tuning_data.ods
    "모델 설정" 시트와 동일한 컬럼의 1행을 만든다.

    파일(adapter_config.json / training_args.bin)만으로는
    판단할 수 없는 항목(비전/언어 레이어, 이미지 사이즈)은
    빈칸으로 남긴다.
    """

    adapter_config, training_args = load_lora_training_config(
        lora_path
    )

    target_modules = set(
        adapter_config.get("target_modules") or []
    )

    attention_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj",
    }
    mlp_modules = {
        "gate_proj", "up_proj", "down_proj",
    }

    has_attention = bool(target_modules & attention_modules)
    has_mlp = bool(target_modules & mlp_modules)

    max_steps = training_args.get("max_steps")
    num_train_epochs = training_args.get("num_train_epochs")

    if max_steps and max_steps > 0:
        step_or_epoch = "step"
        epoch_value = 0
    else:
        step_or_epoch = "epoch" if training_args else ""
        epoch_value = (
            num_train_epochs
            if num_train_epochs is not None
            else ""
        )

    gradient_checkpointing = training_args.get(
        "gradient_checkpointing"
    )

    assistant_only_loss = training_args.get(
        "assistant_only_loss"
    )

    row = {
        "test_name": test_name,

        "모델": base_model.split("/")[-1],

        "방법": "LoRA",

        "데이터셋": dataset_name.split("/")[-1],

        "스탭/에포크": step_or_epoch,

        "컨텍스트 길이":
            training_args.get("max_length", max_seq_length),

        "학습률": training_args.get("learning_rate", ""),

        "랭크": adapter_config.get("r", ""),

        "알파": adapter_config.get("lora_alpha", ""),

        "드롭아웃": adapter_config.get("lora_dropout", ""),

        # 모델을 실제로 로드해 vision/language 어느 쪽에
        # 적용됐는지 확인하기 전에는 확정할 수 없어 비움
        "비전 레이어": "",
        "언어 레이어": "",

        "어텐션 모듈":
            "y" if has_attention
            else ("n" if target_modules else ""),

        "MLP모듈":
            "y" if has_mlp
            else ("n" if target_modules else ""),

        "LoRA종류":
            "LoRA"
            if adapter_config.get("peft_type") == "LORA"
            else adapter_config.get("peft_type", ""),

        "옵티마이저":
            format_optimizer_name(
                training_args.get("optim")
            ),

        "LR 스케줄러":
            str(
                training_args.get("lr_scheduler_type", "")
            ).capitalize(),

        "배치 크기":
            training_args.get(
                "per_device_train_batch_size", ""
            ),

        "그래디언트 누적":
            training_args.get(
                "gradient_accumulation_steps", ""
            ),

        "가중치 감쇠": training_args.get("weight_decay", ""),

        "워밍업 스텝": training_args.get("warmup_steps", ""),

        "에포크": epoch_value,

        "시드": training_args.get("seed", ""),

        # Studio 설정 어디에도 저장되지 않아 비움
        "이미지 사이즈": "",

        "그래디언트 체크포인팅":
            "Unsloth" if gradient_checkpointing else "",

        "어시스턴트 응답만 학습":
            "y" if assistant_only_loss
            else ("n" if assistant_only_loss is not None else ""),
    }

    return pd.DataFrame(
        [row],
        columns=columns,
    )


def save_model_config_log(new_row, path, columns):
    """
    path에 학습 설정을 누적 기록한다.
    같은 test_name이 이미 있으면 덮어쓰고, 없으면 새로 추가한다.
    """

    if os.path.exists(path):

        existing = pd.read_csv(path)

        existing = existing[
            ~existing["test_name"].isin(
                new_row["test_name"]
            )
        ]

        combined = pd.concat(
            [existing, new_row],
            ignore_index=True,
        )

    else:
        combined = new_row

    combined = combined[columns]

    combined.to_csv(path, index=False)

    return combined
