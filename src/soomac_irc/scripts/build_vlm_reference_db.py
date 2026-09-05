#!/usr/bin/env python3

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


# 기존 v11 데이터와 Chroma DB는 덮어쓰지 않고 v13을 별도 경로에 만든다.
DATA_ROOT = Path("/home/roma/ros2_ws/src/soomac_irc/vlm_data/mealkit_coco_v13")
CHROMA_PATH = Path("/home/roma/ros2_ws/src/soomac_irc/vlm_data/chroma_refs_v13")
MANIFEST_PATH = Path("/home/roma/ros2_ws/src/soomac_irc/vlm_data/chroma_refs_v13_manifest.json")

COLLECTION_NAME = "mealkit_refs"
EMBED_MODEL = "google/siglip2-so400m-patch14-384"
SPLIT = "train"

# 클래스별 첫 장면에 치우치지 않도록 고정 seed로 최대 200개를 고른다.
# seed가 같으면 중단 후 다시 실행해도 같은 annotation이 선택된다.
MAX_PER_CLASS = 200
RANDOM_SEED = 20260906
BATCH_SIZE = 32
MIN_TOP3_ACCURACY = 0.90

EXPECTED_CLASSES = {
    "cheese", "crab", "mushroom", "noodle", "onion",
    "pepperoncino", "sauce_cream", "sauce_oil", "sauce_tomato", "sausage",
}


def load_coco():
    annotation_path = DATA_ROOT / SPLIT / "_annotations.coco.json"

    with annotation_path.open(encoding="utf-8") as file:
        coco = json.load(file)

    categories = {category["id"]: category["name"] for category in coco["categories"]}
    images = {image["id"]: image for image in coco["images"]}

    return categories, images, coco["annotations"]


def select_annotations(categories, images, annotations):
    annotations_by_class_and_source = defaultdict(lambda: defaultdict(list))

    for annotation in annotations:
        class_name = categories[annotation["category_id"]]

        if class_name in EXPECTED_CLASSES:
            image = images[annotation["image_id"]]
            source_name = image.get("extra", {}).get("name", image["file_name"])
            annotations_by_class_and_source[class_name][source_name].append(annotation)

    actual_classes = set(annotations_by_class_and_source)

    if actual_classes != EXPECTED_CLASSES:
        missing = sorted(EXPECTED_CLASSES - actual_classes)
        unexpected = sorted(actual_classes - EXPECTED_CLASSES)
        raise ValueError(f"COCO class 불일치 : missing={missing}, unexpected={unexpected}")

    random_generator = random.Random(RANDOM_SEED)
    selected = []

    for class_name in sorted(EXPECTED_CLASSES):
        annotations_by_source = annotations_by_class_and_source[class_name]
        source_names = sorted(annotations_by_source)
        sample_size = min(MAX_PER_CLASS, len(source_names))
        selected_sources = random_generator.sample(source_names, sample_size)

        for source_name in selected_sources:
            annotation = random_generator.choice(annotations_by_source[source_name])
            selected.append((class_name, annotation))

    return selected


def crop_annotation(images, annotation):
    image_path = DATA_ROOT / SPLIT / images[annotation["image_id"]]["file_name"]
    x, y, width, height = annotation["bbox"]

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(image.width, int(x + width + 0.9999))
        bottom = min(image.height, int(y + height + 0.9999))

        if right <= left or bottom <= top:
            raise ValueError(f"bbox 크기 오류 : annotation={annotation['id']}, bbox={annotation['bbox']}")

        return image.crop((left, top, right, bottom))


def load_embedder():
    processor = AutoProcessor.from_pretrained(EMBED_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(
        EMBED_MODEL,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda").eval()

    return model, processor


@torch.no_grad()
def embed_images(model, processor, images):
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    output = model.get_image_features(**inputs)

    if torch.is_tensor(output):
        embeddings = output
    elif getattr(output, "pooler_output", None) is not None:
        embeddings = output.pooler_output
    else:
        embeddings = output.last_hidden_state.mean(dim=1)

    return torch.nn.functional.normalize(embeddings.float(), dim=-1)


def build_collection(model, processor, images, selected):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBED_MODEL, "dataset": "Meal kit.v13i.coco"},
    )

    expected_count = len(selected)

    if collection.count() > expected_count:
        raise ValueError(
            f"기존 v13 Chroma가 현재 설정과 다름 : count={collection.count()}, expected={expected_count}. "
            "기존 DB를 지우지 말고 새 버전 경로를 사용해야 함"
        )

    for start in range(0, expected_count, BATCH_SIZE):
        batch = selected[start:start + BATCH_SIZE]
        crop_images = []
        ids = []
        metadatas = []

        for class_name, annotation in batch:
            crop_images.append(crop_annotation(images, annotation))
            ids.append(f"{SPLIT}_{annotation['id']}")
            metadatas.append({
                "ingredient": class_name,
                "annotation_id": int(annotation["id"]),
                "image_id": int(annotation["image_id"]),
            })

        embeddings = embed_images(model, processor, crop_images).cpu().tolist()
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"임베딩 저장 : {min(start + len(batch), expected_count)}/{expected_count}")

    if collection.count() != expected_count:
        raise ValueError(f"Chroma 개수 오류 : count={collection.count()}, expected={expected_count}")

    return collection


def validate_collection(model, processor, collection):
    annotation_path = DATA_ROOT / "valid" / "_annotations.coco.json"

    with annotation_path.open(encoding="utf-8") as file:
        coco = json.load(file)

    categories = {category["id"]: category["name"] for category in coco["categories"]}
    images = {image["id"]: image for image in coco["images"]}
    checked = Counter()
    passed = 0
    total = 0
    failures = []

    for annotation in coco["annotations"]:
        class_name = categories[annotation["category_id"]]

        if class_name not in EXPECTED_CLASSES or checked[class_name] >= 3:
            continue

        checked[class_name] += 1
        image_path = DATA_ROOT / "valid" / images[annotation["image_id"]]["file_name"]
        x, y, width, height = annotation["bbox"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = image.crop((x, y, x + width, y + height))

        query_embedding = embed_images(model, processor, [crop]).cpu().tolist()
        result = collection.query(query_embeddings=query_embedding, n_results=3)
        top_classes = [metadata["ingredient"] for metadata in result["metadatas"][0]]
        success = top_classes.count(class_name) >= 2
        passed += int(success)
        total += 1
        print(
            f"검증 {class_name}: annotation={annotation['id']}, "
            f"top3={top_classes}, result={'PASS' if success else 'FAIL'}"
        )

        if not success:
            failures.append({
                "annotation_id": int(annotation["id"]),
                "expected": class_name,
                "top3": top_classes,
            })

    accuracy = passed / total if total else 0.0
    return passed, total, accuracy, failures


def main():
    categories, images, annotations = load_coco()
    selected = select_annotations(categories, images, annotations)
    selected_counts = Counter(class_name for class_name, _annotation in selected)

    print("선택 분포 :", dict(sorted(selected_counts.items())))
    print("총 임베딩 :", len(selected))
    print("SigLIP2 로딩")

    model, processor = load_embedder()
    collection = build_collection(model, processor, images, selected)
    passed, total, accuracy, failures = validate_collection(model, processor, collection)

    manifest = {
        "dataset": "Meal kit.v13i.coco",
        "data_root": str(DATA_ROOT),
        "chroma_path": str(CHROMA_PATH),
        "collection": COLLECTION_NAME,
        "embedding_model": EMBED_MODEL,
        "split": SPLIT,
        "random_seed": RANDOM_SEED,
        "max_per_class": MAX_PER_CLASS,
        "count": collection.count(),
        "class_counts": dict(sorted(selected_counts.items())),
        "validation": {
            "passed": passed,
            "total": total,
            "top3_majority_accuracy": accuracy,
            "minimum_accuracy": MIN_TOP3_ACCURACY,
            "failures": failures,
        },
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"검색 검증 : {passed}/{total} ({accuracy:.1%})")
    print(f"manifest : {MANIFEST_PATH}")

    if accuracy < MIN_TOP3_ACCURACY:
        print(f"FAIL: v13 검색 검증 / {accuracy:.1%} < {MIN_TOP3_ACCURACY:.0%} / 실패 crop 확인")
        raise SystemExit(1)

    print("PASS: Meal kit v13 crop 임베딩 Chroma 생성")


if __name__ == "__main__":
    main()
