import os
import re
import gc
import json
import time
import math
import shutil
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from datasets import load_dataset

from unsloth import FastVisionModel


# ============================================================
# GPU 메모리 정리
# ============================================================

def clear_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# LoRA / QLoRA 프로젝트 폴더 탐색
# ============================================================

def resolve_lora_path(
    search_dir,
    project_name,
    prefer_latest_valid=True,
):
    """
    Docker/서버의 Unsloth Studio outputs에서 프로젝트 폴더를 찾는다.

    예:
      search_dir =
        /root/.unsloth/studio/outputs

      project_name =
        project-chartqa-gemma12b-l001

      실제 폴더 =
        unsloth_gemma-3-12b-it__project-chartqa-gemma12b-l001_1788936605

    prefer_latest_valid=True:
      동일 프로젝트 폴더가 여러 개 있어도
      adapter_config.json + adapter_model.safetensors가 존재하는
      "완성된" 폴더만 후보로 삼고, serial이 가장 큰 폴더를 선택한다.
    """

    if not os.path.isdir(search_dir):
        raise FileNotFoundError(
            f"LoRA 검색 폴더가 없습니다: {search_dir}"
        )

    pattern = re.compile(
        re.escape(project_name) + r"_(\d+)$"
    )

    all_candidates = []
    valid_candidates = []

    for name in sorted(os.listdir(search_dir)):
        full_path = os.path.join(search_dir, name)

        if not os.path.isdir(full_path):
            continue

        match = pattern.search(name)

        if not match:
            continue

        serial = match.group(1)

        all_candidates.append(
            (name, serial, full_path)
        )

        adapter_config = os.path.join(
            full_path,
            "adapter_config.json",
        )
        adapter_weights = os.path.join(
            full_path,
            "adapter_model.safetensors",
        )

        if (
            os.path.isfile(adapter_config)
            and os.path.isfile(adapter_weights)
        ):
            valid_candidates.append(
                (name, serial, full_path)
            )

    if not all_candidates:
        raise FileNotFoundError(
            f"'{project_name}' 프로젝트를 "
            f"{search_dir} 에서 찾지 못했습니다."
        )

    if prefer_latest_valid:
        if not valid_candidates:
            listing = "\n".join(
                f"  - {name} (serial={serial})"
                for name, serial, _ in all_candidates
            )
            raise FileNotFoundError(
                "프로젝트 폴더는 찾았지만 완성된 adapter 파일이 없습니다.\n"
                "발견된 폴더:\n"
                + listing
            )

        selected = max(
            valid_candidates,
            key=lambda x: int(x[1]),
        )

        if len(all_candidates) > 1:
            print(
                f"[LoRA 탐색] 동일 프로젝트 폴더 {len(all_candidates)}개 발견"
            )
            print(
                "[LoRA 탐색] 완성된 adapter 중 최신 폴더 자동 선택:",
                selected[0],
            )

        return selected[2], selected[1]

    if len(all_candidates) != 1:
        listing = "\n".join(
            f"  - {name} (serial={serial})"
            for name, serial, _ in all_candidates
        )
        raise RuntimeError(
            f"후보가 {len(all_candidates)}개입니다.\n{listing}"
        )

    return all_candidates[0][2], all_candidates[0][1]


# ============================================================
# checkpoint 정리
# ============================================================

def prune_checkpoints_keep_last(
    project_dir,
    dry_run=False,
):
    """
    project_dir 아래 checkpoint-숫자 디렉토리 중
    숫자가 가장 큰 checkpoint 하나만 남기고 나머지를 삭제한다.

    프로젝트 루트의 최종 adapter_model.safetensors는 건드리지 않는다.

    dry_run=True:
      실제 삭제하지 않고 삭제 대상만 출력한다.
    """

    project_dir = Path(project_dir)

    if not project_dir.is_dir():
        raise FileNotFoundError(
            f"프로젝트 폴더가 없습니다: {project_dir}"
        )

    checkpoints = []

    for path in project_dir.iterdir():
        if not path.is_dir():
            continue

        match = re.fullmatch(
            r"checkpoint-(\d+)",
            path.name,
        )

        if match:
            checkpoints.append(
                (int(match.group(1)), path)
            )

    if not checkpoints:
        print("[checkpoint 정리] checkpoint-* 폴더가 없습니다.")
        return None

    checkpoints.sort(key=lambda x: x[0])

    last_step, last_path = checkpoints[-1]
    delete_targets = checkpoints[:-1]

    print(
        f"[checkpoint 정리] 마지막 checkpoint 유지: "
        f"{last_path.name}"
    )

    if not delete_targets:
        print("[checkpoint 정리] 삭제할 이전 checkpoint가 없습니다.")
        return str(last_path)

    print(
        f"[checkpoint 정리] 이전 checkpoint "
        f"{len(delete_targets)}개 {'삭제 예정' if dry_run else '삭제'}"
    )

    for step, path in delete_targets:
        if dry_run:
            print("  [DRY RUN]", path)
        else:
            shutil.rmtree(path)
            print("  삭제:", path.name)

    return str(last_path)


# ============================================================
# 이미지 전처리
# ============================================================

def resize_for_vlm(image, max_side=1024):
    """
    너무 큰 이미지는 비율을 유지하면서 최대 변을 max_side로 축소.
    작은 이미지는 확대하지 않는다.
    """

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    width, height = image.size

    if max(width, height) <= max_side:
        return image

    scale = max_side / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return image.resize(
        (new_width, new_height)
    )


# ============================================================
# ChartQA 데이터셋 로드
# ============================================================

def _as_answer_list(value):
    """
    ChartQA label이 str / list / tuple 어떤 형태여도
    list[str]로 통일.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]

    return [str(value)]


def load_chartqa_dataset(
    dataset_name,
    split="test",
    limit=30,
    seed=3407,
):
    """
    HuggingFaceM4/ChartQA를 서버에서 직접 로드한다.

    반환 sample:
      image
      image_id
      question
      answers
      human_or_machine

    limit=None이면 split 전체를 사용.
    """

    print(
        f"[Dataset] {dataset_name} / split={split} 로딩"
    )

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    if limit is not None:
        limit = min(int(limit), len(dataset))

        dataset = (
            dataset
            .shuffle(seed=seed)
            .select(range(limit))
        )

    samples = []

    for i, row in enumerate(dataset):
        image = row["image"]

        question = (
            row.get("query")
            or row.get("question")
            or row.get("problem")
            or ""
        )

        raw_answer = (
            row.get("label")
            if "label" in row
            else row.get("answer")
        )

        answers = _as_answer_list(
            raw_answer
        )

        samples.append({
            "image": image,
            "image_id": row.get("id", i),
            "question": str(question),
            "answers": answers,
            "human_or_machine": row.get(
                "human_or_machine",
                "",
            ),
        })

    return samples


# ============================================================
# ChartQA Accuracy
# ============================================================

def normalize_answer(text):
    text = str(text).strip().lower()

    # 모델이 문장으로 답한 경우 앞뒤 불필요 공백 완화
    text = re.sub(r"\s+", " ", text)

    # 마지막 마침표 하나는 제거
    text = text.rstrip(".")

    return text


def parse_numeric_answer(text):
    """
    $, %, comma 등이 포함된 단일 숫자 답변을 숫자로 변환.
    변환 불가하면 None.
    """
    text = normalize_answer(text)

    # 답변이 "42%"인 경우 42로 비교
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("€", "")
    text = text.replace("£", "")
    text = text.strip()

    # percent 기호는 값 자체(42)를 유지
    if text.endswith("%"):
        text = text[:-1].strip()

    try:
        value = float(text)

        if math.isfinite(value):
            return value

    except ValueError:
        return None

    return None


def exact_match_score(prediction, answers):
    pred = normalize_answer(prediction)

    for answer in answers:
        if pred == normalize_answer(answer):
            return 1.0

    return 0.0


def relaxed_accuracy_score(
    prediction,
    answers,
    tolerance=0.05,
):
    """
    ChartQA식 relaxed accuracy:
    - 문자열이면 normalized exact match
    - 숫자이면 GT 대비 5% 이내 오차를 정답으로 처리
    """

    if exact_match_score(prediction, answers) == 1.0:
        return 1.0

    pred_num = parse_numeric_answer(
        prediction
    )

    if pred_num is None:
        return 0.0

    for answer in answers:
        gt_num = parse_numeric_answer(answer)

        if gt_num is None:
            continue

        if gt_num == 0:
            if abs(pred_num - gt_num) <= tolerance:
                return 1.0
        else:
            relative_error = (
                abs(pred_num - gt_num)
                / abs(gt_num)
            )

            if relative_error <= tolerance:
                return 1.0

    return 0.0


def lexical_metrics(text):
    words = re.findall(
        r"[A-Za-z0-9'.%+-]+",
        str(text),
    )

    return len(words)


# ============================================================
# 모델 로드
# ============================================================

def load_model(
    model_path,
    model_name,
    max_seq_length,
    load_in_4bit=False,
):
    print(
        f"\n[{model_name}] 모델 로딩: {model_path}"
    )
    print(
        f"[{model_name}] load_in_4bit={load_in_4bit}"
    )

    clear_gpu()

    model, processor = (
        FastVisionModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,

            # Gemma 3 + Blackwell 환경에서 Flex/Flash Attention이
            # 각각 Triton shared-memory / fake-tensor storage 오류를
            # 일으킬 수 있어 평가에서는 안정적인 eager backend를 사용.
            attn_implementation="eager",

            # CUDA_VISIBLE_DEVICES=0으로 한 장만 보이지만
            # device placement도 명시적으로 고정한다.
            device_map={"": 0},
        )
    )

    FastVisionModel.for_inference(model)

    return model, processor


# ============================================================
# VLM 입력
# ============================================================

def make_inputs(
    processor,
    image,
    question,
):
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
                    "text": question,
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
    question,
    max_new_tokens,
):
    inputs = make_inputs(
        processor,
        image,
        question,
    )

    input_length = (
        inputs["input_ids"].shape[1]
    )
    input_tokens = input_length

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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

    output_tokens = generated.shape[1]

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
        else 0.0
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

def warmup(
    model,
    processor,
    dataset,
):
    if not dataset:
        raise ValueError(
            "평가 데이터셋이 비어 있습니다."
        )

    sample = dataset[0]

    inputs = make_inputs(
        processor,
        sample["image"],
        sample["question"],
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()

    del inputs
    del output

    clear_gpu()


# ============================================================
# 모델 전체 평가
# ============================================================

def evaluate_model(
    model_path,
    model_name,
    dataset,
    save_path,
    max_seq_length,
    max_new_tokens,
    test_name,
    load_in_4bit=False,
):
    num_samples = len(dataset)

    model, processor = load_model(
        model_path=model_path,
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    warmup(
        model,
        processor,
        dataset,
    )

    results = []

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[{model_name}] {num_samples}개 ChartQA 추론 시작"
    )

    for i, sample in enumerate(dataset):
        result = inference_one(
            model=model,
            processor=processor,
            image=sample["image"],
            question=sample["question"],
            max_new_tokens=max_new_tokens,
        )

        answers = sample["answers"]

        exact = exact_match_score(
            result["prediction"],
            answers,
        )

        relaxed = relaxed_accuracy_score(
            result["prediction"],
            answers,
        )

        row = {
            "sample_index": i,
            "image_id": sample.get(
                "image_id",
                "",
            ),
            "test_name": test_name,
            "question": sample["question"],
            "gt_answers": json.dumps(
                answers,
                ensure_ascii=False,
            ),
            "prediction": result["prediction"],
            "exact_match": exact,
            "relaxed_accuracy": relaxed,
            "latency_sec": result["latency_sec"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "tokens_per_sec": result["tokens_per_sec"],
            "peak_vram_gb": result["peak_vram_gb"],
            "word_count": lexical_metrics(
                result["prediction"]
            ),
            "human_or_machine": sample.get(
                "human_or_machine",
                "",
            ),
        }

        results.append(row)

        pd.DataFrame(
            results
        ).to_csv(
            save_path,
            index=False,
        )

        print(
            f"[{model_name}] "
            f"{i + 1:04d}/{num_samples} | "
            f"exact={exact:.0f} | "
            f"relaxed={relaxed:.0f} | "
            f"{result['latency_sec']:.2f}s | "
            f"{result['tokens_per_sec']:.2f} tok/s | "
            f"{result['peak_vram_gb']:.2f} GB"
        )

    del model
    del processor

    clear_gpu()

    return results


# ============================================================
# 최종 평균
# ============================================================

def make_summary(df):
    metrics = [
        "exact_match",
        "relaxed_accuracy",
        "latency_sec",
        "tokens_per_sec",
        "peak_vram_gb",
        "output_tokens",
        "word_count",
    ]

    return (
        df
        .groupby("test_name")[metrics]
        .mean(numeric_only=True)
        .round(4)
    )


# ============================================================
# 결과 로그
# ============================================================

def build_results_log_rows(
    df,
    columns,
):
    rows = []

    for test_name, group in df.groupby(
        "test_name"
    ):
        rows.append({
            "test_name":
                test_name,

            "Exact Accuracy":
                round(
                    group["exact_match"].mean(),
                    4,
                ),

            "Relaxed Accuracy":
                round(
                    group["relaxed_accuracy"].mean(),
                    4,
                ),

            "AVG Latency":
                f"{group['latency_sec'].mean():.2f} s",

            "Tokens/sec":
                round(
                    group["tokens_per_sec"].mean(),
                    2,
                ),

            "Peak VRAM":
                f"{group['peak_vram_gb'].mean():.2f} GB",

            "AVG Output Tokens":
                round(
                    group["output_tokens"].mean(),
                    1,
                ),

            "AVG length":
                round(
                    group["word_count"].mean(),
                    1,
                ),
        })

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def save_results_log(
    new_rows,
    path,
    columns,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
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
    combined.to_csv(
        path,
        index=False,
    )

    return combined


# ============================================================
# 학습 설정 로드
# ============================================================

def load_lora_training_config(
    lora_path,
):
    with open(
        os.path.join(
            lora_path,
            "adapter_config.json",
        )
    ) as f:
        adapter_config = json.load(f)

    training_args = {}

    training_args_path = os.path.join(
        lora_path,
        "training_args.bin",
    )

    if os.path.exists(training_args_path):
        try:
            args_obj = torch.load(
                training_args_path,
                weights_only=False,
            )

            if hasattr(args_obj, "to_dict"):
                training_args = args_obj.to_dict()
            elif isinstance(args_obj, dict):
                training_args = args_obj

        except Exception as e:
            print(
                "training_args.bin 읽기 실패 "
                "(일부 설정은 빈칸):",
                e,
            )

    return (
        adapter_config,
        training_args,
    )


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

    return mapping.get(
        str(optim),
        str(optim),
    )


def build_model_config_row(
    lora_path,
    test_name,
    base_model,
    dataset_name,
    max_seq_length,
    columns,
    fine_tune_method="LoRA",
):
    adapter_config, training_args = (
        load_lora_training_config(
            lora_path
        )
    )

    target_modules = set(
        adapter_config.get(
            "target_modules"
        )
        or []
    )

    attention_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }

    mlp_modules = {
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    has_attention = bool(
        target_modules
        & attention_modules
    )

    has_mlp = bool(
        target_modules
        & mlp_modules
    )

    max_steps = training_args.get(
        "max_steps"
    )

    num_train_epochs = training_args.get(
        "num_train_epochs"
    )

    if (
        max_steps is not None
        and max_steps > 0
    ):
        step_or_epoch = "step"
        epoch_value = 0
    else:
        step_or_epoch = (
            "epoch"
            if training_args
            else ""
        )
        epoch_value = (
            num_train_epochs
            if num_train_epochs is not None
            else ""
        )

    gradient_checkpointing = (
        training_args.get(
            "gradient_checkpointing"
        )
    )

    assistant_only_loss = (
        training_args.get(
            "assistant_only_loss"
        )
    )

    row = {
        "test_name":
            test_name,

        "모델":
            base_model.split("/")[-1],

        "방법":
            fine_tune_method,

        "데이터셋":
            dataset_name.split("/")[-1],

        "스탭/에포크":
            step_or_epoch,

        "컨텍스트 길이":
            training_args.get(
                "max_length",
                max_seq_length,
            ),

        "학습률":
            training_args.get(
                "learning_rate",
                "",
            ),

        "랭크":
            adapter_config.get(
                "r",
                "",
            ),

        "알파":
            adapter_config.get(
                "lora_alpha",
                "",
            ),

        "드롭아웃":
            adapter_config.get(
                "lora_dropout",
                "",
            ),

        # adapter/training_args만으로는 vision/language
        # 적용 여부를 확정하기 어려워 빈칸 유지.
        "비전 레이어": "",
        "언어 레이어": "",

        "어텐션 모듈":
            "y"
            if has_attention
            else (
                "n"
                if target_modules
                else ""
            ),

        "MLP모듈":
            "y"
            if has_mlp
            else (
                "n"
                if target_modules
                else ""
            ),

        "LoRA종류":
            (
                "LoRA"
                if adapter_config.get(
                    "peft_type"
                ) == "LORA"
                else adapter_config.get(
                    "peft_type",
                    "",
                )
            ),

        "옵티마이저":
            format_optimizer_name(
                training_args.get(
                    "optim"
                )
            ),

        "LR 스케줄러":
            str(
                training_args.get(
                    "lr_scheduler_type",
                    "",
                )
            ).capitalize(),

        "배치 크기":
            training_args.get(
                "per_device_train_batch_size",
                "",
            ),

        "그래디언트 누적":
            training_args.get(
                "gradient_accumulation_steps",
                "",
            ),

        "가중치 감쇠":
            training_args.get(
                "weight_decay",
                "",
            ),

        "워밍업 스텝":
            training_args.get(
                "warmup_steps",
                "",
            ),

        "에포크":
            epoch_value,

        "시드":
            training_args.get(
                "seed",
                "",
            ),

        "이미지 사이즈":
            "",

        "그래디언트 체크포인팅":
            (
                "Unsloth"
                if gradient_checkpointing
                else ""
            ),

        "어시스턴트 응답만 학습":
            (
                "y"
                if assistant_only_loss
                else (
                    "n"
                    if assistant_only_loss
                    is not None
                    else ""
                )
            ),
    }

    return pd.DataFrame(
        [row],
        columns=columns,
    )


def save_model_config_log(
    new_row,
    path,
    columns,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
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
    combined.to_csv(
        path,
        index=False,
    )

    return combined
