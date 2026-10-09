#!/home/roma/miniconda3/envs/gemma4_env/bin/python

import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

import rclpy
from std_msgs.msg import String

if __package__:
    from .stt_nemotron_node import (
        PCM_FULL_SCALE,
        SAMPLE_RATE,
        NemotronSttNode,
        NemotronSttStats,
    )
else:
    from stt_nemotron_node import (
        PCM_FULL_SCALE,
        SAMPLE_RATE,
        NemotronSttNode,
        NemotronSttStats,
    )


# Nemotron 모델과 같은 /home/roma/models/audio/stt 아래에 로컬 가중치를 둔다.
MODEL_DIR = '/home/roma/models/audio/stt/Qwen3-ASR-1.7B-hf'
COMPUTE_DEVICE = 'auto'
MODEL_DTYPE = torch.bfloat16
# Qwen3-ASR가 자동 감지하지 않고 한국어 디코딩 형식을 쓰도록 고정한다.
LANGUAGE = 'ko'
# VAD가 발화를 30초에서 닫으므로 256토큰이면 주문 문장에 충분하다.
MAX_NEW_TOKENS = 256


class Qwen3AsrSttNode(NemotronSttNode):
    """Nemotron의 ROS 토픽·VAD·게이트 로직에 Qwen3-ASR 추론만 연결한다."""

    def __init__(self):
        super().__init__(
            node_name='soomac_qwen3_asr_stt_node',
            engine_name='Qwen3-ASR')

    def _choose_compute_device(self):
        if COMPUTE_DEVICE == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if COMPUTE_DEVICE == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('그래픽 연산 장치를 지정했지만 CUDA를 사용할 수 없어요.')
        if COMPUTE_DEVICE not in ('cuda', 'cpu'):
            raise ValueError(f'지원하지 않는 연산 장치예요: {COMPUTE_DEVICE}')
        return COMPUTE_DEVICE

    def _load_model(self):
        model_path = Path(MODEL_DIR).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f'모델 디렉터리가 없어요: {model_path}')

        compute_device = self._choose_compute_device()
        weight_files = (
            tuple(model_path.glob('*.safetensors'))
            + tuple(model_path.glob('pytorch_model*.bin')))
        incomplete_files = tuple(
            (model_path / '.cache' / 'huggingface' / 'download').glob(
                '*.incomplete'))
        if not weight_files:
            incomplete_text = (
                f' 불완전 다운로드 {len(incomplete_files)}개가 남아 있어요.'
                if incomplete_files else '')
            raise FileNotFoundError(
                f'Qwen3-ASR 가중치가 없어요: {model_path}.'
                f'{incomplete_text}')

        model_dtype = MODEL_DTYPE if compute_device == 'cuda' else torch.float32
        loading_started_at = time.perf_counter()
        self.get_logger().info(
            f'Qwen3-ASR 모델을 불러옵니다. 경로 {model_path}, '
            f'연산 장치 {compute_device}입니다.')

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True)
            processor_sample_rate = self.processor.feature_extractor.sampling_rate
            if processor_sample_rate != SAMPLE_RATE:
                raise RuntimeError(
                    f'모델 입력은 {processor_sample_rate}헤르츠인데 '
                    f'마이크 설정은 {SAMPLE_RATE}헤르츠예요.')

            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=model_dtype,
                device_map='cuda:0' if compute_device == 'cuda' else 'cpu')
            self.model.eval()

            loading_seconds = time.perf_counter() - loading_started_at
            model_device = self.model.device
            if model_device.type == 'cuda':
                allocated_vram_gib = (
                    torch.cuda.memory_allocated(model_device) / (1024 ** 3))
                reserved_vram_gib = (
                    torch.cuda.memory_reserved(model_device) / (1024 ** 3))
                vram_text = (
                    f'할당 {allocated_vram_gib:.2f}기가바이트, '
                    f'예약 {reserved_vram_gib:.2f}기가바이트')
            else:
                vram_text = '사용 안 함'

            self.get_logger().info(
                f'Qwen3-ASR 모델을 {loading_seconds:.2f}초 만에 불러왔어요. '
                f'자료형 {self.model.dtype}, 장치 {model_device}, '
                f'그래픽 메모리 {vram_text}입니다.')

        except Exception as error:
            self.get_logger().error(
                f'Qwen3-ASR 모델을 불러오다가 터졌어요: '
                f'{error}\n{traceback.format_exc()}')
            raise

    def _recognize_utterance(self, utterance_pcm, captured_session,
                             utterance_ended_at, captured_seconds):
        stats = NemotronSttStats(captured_seconds, utterance_ended_at)

        try:
            if (self.stop_event.is_set()
                    or captured_session != self._current_session()
                    or not self._is_gate_open()):
                return

            processor = self.processor
            model = self.model
            if processor is None or model is None:
                raise RuntimeError('Qwen3-ASR 모델 자원이 이미 정리됐어요.')

            audio = np.frombuffer(
                utterance_pcm, dtype='<i2').astype(np.float32) / PCM_FULL_SCALE

            stats.mark_inference_started()
            inputs = processor.apply_transcription_request(
                audio=audio,
                language=LANGUAGE).to(model.device, model.dtype)
            prompt_token_count = inputs['input_ids'].shape[1]

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS)

            generated_ids = output_ids[:, prompt_token_count:]
            decoded = processor.decode(
                generated_ids,
                return_format='transcription_only')
            raw_text = decoded[0] if isinstance(decoded, list) else decoded
            # 핫워드·메뉴 단어 보정은 연결하지 않고 공백만 정리한다.
            final_text = ' '.join(str(raw_text or '').split())
            stats.mark_inference_finished()

            if not final_text:
                self.get_logger().info('말은 끝났는데 알아들은 글자가 없어요.')
                self.get_logger().info(stats.summary())
                return

            with self.state_lock:
                if (self.stop_event.is_set()
                        or captured_session != self.session
                        or not self.listening_enabled
                        or not self.gate_open):
                    result_is_stale = True
                else:
                    self.question_pub.publish(String(data=final_text))
                    stats.mark_published()
                    self.gate_open = False
                    self._gate_open_at = None
                    result_is_stale = False

            if result_is_stale:
                self.get_logger().info(
                    f'Qwen3-ASR 세션 {captured_session}번 결과는 '
                    '문이 닫혀서 버렸어요.')
                return

            self.get_logger().info(f'질문으로 보냈어요: {final_text}')
            self.get_logger().info(stats.summary())

        except Exception as error:
            self.get_logger().error(
                f'음성 인식하다가 터졌어요: {error}\n{traceback.format_exc()}')


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = Qwen3AsrSttNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(
                f'STT 노드가 터졌어요: {error}\n{traceback.format_exc()}')
        else:
            print(f'STT 노드를 시작하지 못했어요: {error}\n{traceback.format_exc()}')
        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
