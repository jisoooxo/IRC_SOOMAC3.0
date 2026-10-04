import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig

# AutoProcessor : 텍스트, 이미지, 오디오 등의 다양한 입력 데이터를 모델이 이해할 수 있는 형태(숫자 텐서)로 자동 변환하고, 역으로 모델의 출력값을 사람이 읽을 수 있는 형태로 복원하는 역할

try:
    from transformers import AutoModelForMultimodalLM as _AutoVLM
except ImportError:
    # transformers 구버전 호환
    from transformers import AutoModelForImageTextToText as _AutoVLM

# 모델 경로
MODEL_PATH = "/home/roma/Desktop/sLLM/gemma-4-12B-it"
DECISION_ADAPTER_PATH = "/home/roma/IRC_SOOMAC3.0/models/decision_adapter_epoch6_best"
MODEL_QUANTIZATION = "nf4" # int8, nf4(qlora), bf16(원본)

VLM_MAX_TOKENS = 1024

def make_call_vlm(model, processor, logger=None):
    # vlm 호출 함수(model, processor은 변화 X)

    tokenizer = processor.tokenizer 

    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]

    stop_ids = list(dict.fromkeys(
        token_id
        for token_id in stop_ids
        if isinstance(token_id, int) and token_id >= 0
    ))

    @torch.inference_mode() # 추론 모드
    def call_vlm(pil_images: list, system_prompt:str, user_text: str)-> str:
        # 입력은 PIL 이미지 목록, VLM system prompt, 사용자 질문
        # output은 vlm문자열, 호출 함수 자체(호출자)
        if not isinstance(pil_images, list) or not pil_images:
            raise ValueError("VLM 입력 이미지가 없음")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("VLM system prompt가 없음")
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("VLM 사용자 질문이 없음")

        # 위는 굳이 필요 X

        
        content = [] # 이미지랑 프롬프트 담을 그릇

        for image in pil_images:
            content.append({"type": "image", "image": image}) 

        content.append({"type": "text", "text": user_text}) # 내 질문 ㅇㅇ

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # system prompt는 페르소나용, 내 질문은 이미지랑 텍스트 다 list로 묶어서 던져버림

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True, # 텐서까지 만들어서 반환한다(텍스트토큰화와 이미지 전처리를 한번에 할 수 있다). False이면 템플릿이 적용된 문자열만 반환한다(이거 하면 다시 processor 호출해야함.)
            return_dict=True, # input_ids만이 아니라 attention_mask, pixel_values 등을 dict로 반환한다. VLM은 이미지 텐서가 필요하므로 True여야 한다
            return_tensors="pt",
            enable_thinking=False, # thinking 켜지 않음.
        ).to(model.device)
        prompt_length = inputs["input_ids"].shape[1] 
        
        # 프롬프트 길이 기록
        # 이 이유는 나중에 모델이 응답 뱉을 때 input도 토큰을 소비해버려서 잘라버려야함.

        if logger is not None:
            logger.info(f"VLM prompt tokens: {prompt_length}, images: {len(pil_images)}")

        generate_args = {
            **inputs,
            "max_new_tokens": VLM_MAX_TOKENS, # 새로 생성하는 토큰의 최대 개수. 프롬프트 길이는 여기에 포함되지 않는다
            "do_sample": False, # greedy 디코딩: 매 스텝마다 확률이 가장 높은 토큰 1개를 고른다. 같은 입력이면 항상 같은 출력이 나온다. 
            "use_cache": True, # KV Cache 사용. 빨라지나 VRAM에 캐시 더 올려서 vram 많이 먹음
            "eos_token_id": stop_ids, # 이 토큰 중 하나가 나오면 생성을 멈춘다
        }

        """
        do sample은 do_sample: true, temperature 1.0, top_k 64, top_p 0.95로 되어 있다. 그러나 False로 하면 temperature, top p, top k 다 무시 가능.
        """

        """
        autocast -> vision tower, LM 사이 dtype 충돌 방지용

        [문제 상황]
        - processor가 만든 pixel_values는 float32임 (0~255를 0~1로 rescale한 실수값)
        - vision tower는 들어온 pixel_values를 자기 가중치 dtype에 맞춰서 캐스팅하게 짜여있음
          transformers/models/gemma4_unified/modeling_gemma4_unified.py
            826줄: if (target_dtype := self.patch_dense.weight.dtype).is_floating_point:
            827줄:     pixel_values = pixel_values.to(target_dtype)
            873줄: embedding_projection 도 같은 방식
        - 근데 int8 양자화하면 nn.Linear 가중치가 bitsandbytes int8 텐서로 바뀜
          -> weight.dtype == torch.int8 -> is_floating_point 가 False
          -> 캐스팅을 그냥 건너뜀 -> pixel_values가 float32 그대로 흘러감
        - 반면 LayerNorm, pos_embedding 같은 양자화 안 된 부분은 dtype=bf16으로 로드됨
          -> float32 입력 + bf16 가중치가 한 연산에서 만남 -> dtype 안 맞아서 에러 나거나 이상하게 섞임
          (⚠ 정확히 어떤 에러 메시지가 나는지는 직접 돌려서 확인은 안 해봄)

        [autocast가 하는 일]
        - with 블록 안에서 연산할 때만 dtype을 자동으로 맞춰줌. 가중치 자체는 안 바뀜
        - matmul, linear 같은 건 bf16으로 캐스팅해서 계산
        - layer_norm, softmax 같이 정밀도 중요한 건 float32로 올려서 계산
        - 그래서 입력이 float32든 bf16이든 연산마다 dtype이 통일됨 -> 충돌 X

        [양자화 모드별로 필요한지]
        - int8: 필요함. 위에 쓴 대로 weight.dtype이 int8이라 캐스팅이 빠짐
        - nf4: bnb_4bit_quant_storage=bf16 이라 weight.dtype이 bf16으로 보임
               -> 827줄 캐스팅이 정상 동작함 -> 없어도 될 가능성 높음 (quant_storage 기본값 uint8이면 int8이랑 같은 문제)
        - bf16: 원래 다 bf16이라 필요 없음. 켜져 있어도 손해는 거의 없음

        [인자]
        - device_type: "cuda"/"cpu". 어느 장치 연산에 적용할지
        - dtype: 자동 캐스팅 목표 dtype (bf16)
        - enabled: cuda일 때만 켬. cpu면 아무것도 안 함
        """

        autocast_device = model.device.type

        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=autocast_device == "cuda"):
            # adapter(LoRA) 붙어있으면 VLM 호출 동안만 잠깐 끄고 원본 gemma로 추론. with 끝나면 다시 켜짐
            # adapter는 텍스트 판단용으로 학습한 거라 이미지 설명엔 원본이 맞음 -> 모델 하나로 LLM, VLM 둘 다 씀
            if isinstance(model, PeftModel):
                with model.disable_adapter():
                    output = model.generate(**generate_args)
            else:
                output = model.generate(**generate_args)

        reply = processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
        # output에는 프롬프트+답변이 같이 들어있어서 prompt_length 뒤부터만 잘라서 디코딩

        # enable_thinking=False여도 모델이 <|channel>thought<channel|> 흉내내서 뱉는 경우 있음
        # skip_special_tokens가 특수토큰만 지우고 "thought" 글자는 남겨서 여기서 따로 잘라줌
        if reply.startswith("thought"):
            reply = reply[len("thought"):].lstrip()

        if logger is not None:
            logger.info(f"VLM output: {reply}")

        return reply

    return call_vlm        


def load_model(adapter_path=None):
    # gemma4를 한번에 로딩하고, adapter는 차후에 ㅇㅇ

    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    # processor = tokenizer(텍스트) + image processor(이미지) + feature extractor(오디오) 묶음

    """
    from_pretrained 공통 인자
    - device_map={"": 0}: ""는 모델 전체(루트)라는 뜻. 통째로 GPU 0번에 올림
      "auto"랑 달리 CPU로 안 내림 -> VRAM 모자라면 그냥 OOM
    - attn_implementation="sdpa": pytorch 내장 scaled_dot_product_attention 씀
      알아서 flash/mem-efficient 커널 골라줌. 대안은 "eager"(느린데 attention weight 볼 수 있음), "flash_attention_2"(따로 설치해야함)
    - low_cpu_mem_usage=True: transformers 5.13에선 무시됨 (modeling_utils.py 4152줄에서 그냥 버림). 지워도 됨
    - dtype=bf16: 양자화 안 되는 부분(embedding, norm 등) dtype. v5부터 torch_dtype 대신 dtype 씀
    - local_files_only=True: HF hub 접속 안 하고 로컬 경로만 읽음
    """

    if MODEL_QUANTIZATION == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, # Linear 가중치를 int8로 저장(LLM.int8()). bf16 대비 VRAM 절반
            llm_int8_threshold=6.0, # activation 절댓값이 6 넘는 outlier 차원은 int8 말고 16bit로 따로 계산. 낮추면 정확도↑ 속도↓
            llm_int8_has_fp16_weight=False, # int8 가중치만 들고 있음. True면 fp16 원본도 같이 들고 있음(학습용). 추론이면 False
        )
        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            local_files_only=True,
        )

    elif MODEL_QUANTIZATION == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, # 가중치 4bit로 저장. bf16 대비 VRAM 약 1/4
            bnb_4bit_quant_type="nf4", # NormalFloat4. 정규분포 모양인 가중치에 맞춘 4bit 형식이라 "fp4"보다 품질 좋음
            bnb_4bit_use_double_quant=True, # 양자화 scale 상수까지 한번 더 양자화. 파라미터당 약 0.4bit 더 아낌
            bnb_4bit_compute_dtype=torch.bfloat16, # 실제 matmul은 4bit를 bf16으로 풀어서 계산
            bnb_4bit_quant_storage=torch.bfloat16, # 4bit 값들 묶어 담는 저장 텐서 dtype. FSDP/멀티GPU에서 중요. 여기선 weight.dtype이 bf16으로 보이게 해서 vision tower 캐스팅이 정상 동작하는 효과도 있음
        )
        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            local_files_only=True,
        )

    elif MODEL_QUANTIZATION == "bf16":
        # 양자화 없이 원본 그대로. 23GB + KV cache라 VRAM 제일 많이 먹는데 품질, 속도는 제일 좋음
        model = _AutoVLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            local_files_only=True,
        )

    else:
        raise ValueError(f"MODEL_QUANTIZATION은 int8, nf4, bf16 중 하나여야 함: {MODEL_QUANTIZATION}")

    model.eval() # dropout 등 추론 모드로

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False) # LoRA adapter 읽기 전용으로 붙임
        model.eval() # PeftModel로 새로 감쌌으니 한번 더

    return model, processor


"""
gemma4 (gemma-4-12B-it) 구조 정리 (config.json, processor_config.json 기준)

- 클래스: Gemma4UnifiedForConditionalGeneration
  텍스트 LM + vision tower + audio tower가 한 모델에 다 들어있는 멀티모달 모델임.
  그래서 LLM(텍스트 판단)이랑 VLM(이미지 설명)을 모델 하나로 같이 씀.
- 크기: model.safetensors 23GB (bf16 원본 기준. 2byte x 약 12B 파라미터)
  int8이면 대략 절반, nf4면 대략 1/4 수준 + KV cache 만큼 더 먹음.
- MoE 아님(enable_moe_block=False). 전부 dense임.

[텍스트 LM]
- 레이어 48개, hidden 3840, attention head 16개
- sliding window attention 5층 + full attention 1층 이 반복됨 (5:1)
  sliding은 주변 1024토큰만 봄 -> 긴 문맥에서도 KV cache가 덜 커짐
  full은 전체를 다 봄 -> 먼 문맥 정보는 이 층들이 담당
- KV head: sliding층 8개(GQA), full층 1개 + K=V 공유(attention_k_eq_v) -> KV cache 절약용
- 최대 문맥 262144 토큰(256K), vocab 262144
- final_logit_softcapping 30: 마지막 logit을 30 근처로 눌러서 값이 튀지 않게 함

[vision tower]
- 이미지를 16x16 patch로 자르고, 3x3 pooling해서 48x48 단위로 합침
- 이미지 1장당 최대 280 토큰(max_soft_tokens=280)으로 LM에 들어감
  -> 이미지 여러장 넣으면 프롬프트 토큰이 280씩 늘어남 (로그의 VLM prompt tokens 참고)
- 이미지 전처리: 0~255 -> 0~1로 rescale만 하고 mean/std normalize는 안 함
- use_bidirectional_attention="vision": 이미지 토큰끼리는 앞뒤 양방향으로 봄 (텍스트는 원래대로 앞만 봄)

[채팅 포맷]
- 한 턴은 <|turn> ... <turn|> 으로 감쌈. 그래서 <turn|> 가 나오면 답변 끝난 거라 stop_ids에 넣음
- thinking은 <|channel>thought ... <channel|> 채널로 나옴
"""