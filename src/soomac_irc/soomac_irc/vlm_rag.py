import json
import chromadb
import numpy as np
from PIL import Image


DATA_ROOT = "/home/roma/ros2_ws/src/soomac_irc/vlm_data/mealkit_coco_v11"
CHROMA_PATH = "/home/roma/ros2_ws/src/soomac_irc/vlm_data/chroma_refs_v11"


class VlmRag:
    def __init__(self):

        # Chroma DB에는 이미지가 아니라 임베딩, annotation id가 저장되어 있다.

        self.collection = chromadb.PersistentClient(path=CHROMA_PATH,).get_collection("mealkit_refs")

        # COCO annotation JSON을 읽어 이미지 파일명과 bbox 정보를 가져옴

        with open(f"{DATA_ROOT}/train/_annotations.coco.json", encoding="utf-8",) as file:
            coco = json.load(file) # JSON 파일 내용을 Python 딕셔너리로 변환


        images = {} # 이미지 아이디로 실제 이미지 파일 이름을 찾기 위한 변수

        for image in coco["images"]:
            images[image["id"]] = image["file_name"]


        self.annotations = {} # annotation id로 이미지 파일과 bbox를 찾기 위한 변수

        for annotation in coco["annotations"]:
            image_id = annotation["image_id"]
            filename = images[image_id]

            self.annotations[annotation["id"]] = (filename, annotation["bbox"],)



    def get_reference(self, ingredient):
        # Chroma metadata는 영어 재료명으로 되어있음

        if ingredient == "양파":
            name = "onion"

        elif ingredient == "버섯":
            name = "mushroom"

        elif ingredient == "소시지":
            name = "sausage"

        elif ingredient == "게살":
            name = "crab"

        elif ingredient in ("얇은면", "넓적면"):
            name = "noodle"

        elif ingredient == "크림":
            name = "sauce_cream"

        elif ingredient == "오일":
            name = "sauce_oil"

        elif ingredient == "토마토":
            name = "sauce_tomato"

        else:
            return None

        result = self.collection.get(where={"ingredient": name}, include=["embeddings"])

        if not result["ids"] or len(result["embeddings"]) == 0:
            return None

        # 같은 재료 이미지들의 임베딩 중심을 먼저 구함

        """
        data
        docs
        embeddings
        ids
        included
        metadatas
        uris
        """

        embeddings = np.asarray(result["embeddings"], dtype=np.float32)
        center = embeddings.mean(axis=0)
        center_norm = np.linalg.norm(center) # 중심 벡터의 길이

        if center_norm == 0:
            return None

        center = center / center_norm

        # 유사도
        # 저장된 임베딩과 중심 벡터가 정규화되어 있어서 내적값이 코사인 유사도가 됨
        similarities = embeddings @ center # 각 참고 이미지와 대표 중심이 얼마나 비슷한가

        reference_index = int(similarities.argmax())


        chroma_id = result["ids"][reference_index] # 아이디 빼서 
        annotation_id = int(chroma_id.split("_", 1)[1]) # annotation id 잘라서 가져옴

        annotation = self.annotations.get(annotation_id)

        if annotation is None:
            return None

        filename, bbox = annotation
        x, y, width, height = bbox

        with Image.open(f"{DATA_ROOT}/train/{filename}") as image:
            image = image.convert("RGB")

            return image.crop((x, y, x+width, y+height))

