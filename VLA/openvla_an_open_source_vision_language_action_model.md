# OpenVLA: An Open-Source Vision-Language-Action Model

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn

## 🧩 Problem to Solve

로봇 조작을 위한 학습된 정책들은 훈련 데이터 범위를 넘어서는 일반화 능력(예: 방해물, 새로운 객체, 미확인 지시에 대한 견고성)이 부족합니다. 기존 Vision-Language-Action (VLA) 모델들은 대부분 비공개이며 접근하기 어렵고, 새로운 작업에 대한 VLA의 효율적인 미세 조정 방법이 제대로 탐구되지 않아 널리 채택되기 어렵습니다. 따라서 오픈 소스 대규모 언어 모델(LLM) 생태계와 유사하게, 효과적인 미세 조정 및 적응을 지원하는 오픈 소스 범용 VLA 모델의 필요성이 대두됩니다.

## ✨ Key Contributions

- **OpenVLA 공개**: Open X-Embodiment 데이터셋의 970k 로봇 에피소드로 훈련된 7B 매개변수 오픈 소스 VLA 모델인 OpenVLA를 소개합니다.
- **최고 성능 달성**: 범용 로봇 조작 정책에서 새로운 최첨단 성능을 달성했으며, 55B 매개변수의 폐쇄형 모델인 RT-2-X를 7배 적은 매개변수로도 29개 작업에서 절대 성공률 16.5% 더 높은 성능을 보였습니다.
- **다중 로봇 제어 및 효율적인 미세 조정**: 여러 로봇을 즉시 제어할 수 있으며, 매개변수 효율적인 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 및 4비트 양자화(quantization)를 통해 일반 소비자용 GPU에서도 새로운 로봇 도메인에 빠르게 적응할 수 있음을 입증했습니다.
- **언어 기반 작업 일반화**: 언어 기반(language grounding)이 중요한 다중 객체, 다중 지시 환경에서 처음부터 모방 학습(imitation learning)하는 Diffusion Policy보다 20.4% 더 나은 성능을 보여주었습니다.
- **자원 공개**: 모델 체크포인트, 미세 조정 노트북, PyTorch 학습 코드베이스를 전면 오픈 소스화하여 향후 VLA 연구 개발을 촉진합니다.

## 📎 Related Works

- **Visually-Conditioned Language Models (VLM)**: CLIP, SigLIP, Llama 2, PaLI 등 인터넷 규모의 데이터로 사전 훈련된 VLM을 활용합니다. 특히, DINOv2와 SigLIP의 사전 훈련된 특징을 융합하여 다중 해상도 시각 특징을 사용하는 "patch-as-token" 아키텍처를 채택했습니다. OpenVLA의 백본으로는 Prismatic-7B VLM [44]을 사용합니다.
- **Generalist Robot Policies**: Octo [5], RT-1 [2] 등 대규모 로봇 데이터셋으로 훈련된 다중 작업 범용 로봇 정책이 있습니다. 이들은 일반적으로 사전 훈련된 구성 요소에 스크래치에서 초기화된 추가 구성 요소를 결합하는 반면, OpenVLA는 VLM을 직접 미세 조정하여 로봇 행동을 생성하는 엔드투엔드(end-to-end) 방식을 채택합니다.
- **Vision-Language-Action Models (VLA)**: RT-2 [7], RT-2-X [1] 등 대규모 사전 훈련된 VLM을 로봇 행동 예측에 직접 미세 조정하는 연구가 있었습니다. 이러한 모델들은 강력한 일반화 능력을 보였으나, 대부분 비공개이며 새로운 로봇 설정에 대한 효율적인 미세 조정을 지원하지 않는다는 한계가 있습니다. OpenVLA는 이러한 한계를 극복하고 오픈 소스 VLA를 제공합니다.

## 🛠️ Methodology

OpenVLA는 Prismatic-7B VLM 백본을 로봇 행동 예측을 위해 미세 조정하여 개발되었습니다.

1. **모델 아키텍처**:
   - **시각 인코더(Vision Encoder)**: DINOv2 [25]와 SigLIP [79]에서 사전 훈련된 특징을 채널 단위로 연결하여 융합합니다. 이는 향상된 공간 추론(spatial reasoning) 능력을 제공합니다.
   - **프로젝터(Projector)**: 시각 인코더의 출력 임베딩을 언어 모델의 입력 공간으로 매핑하는 소규모 2계층 MLP입니다.
   - **LLM 백본(LLM Backbone)**: Llama 2 7B 대규모 언어 모델 [10]입니다.
2. **훈련 절차**:
   - **액션 이산화(Action Discretization)**: 연속적인 로봇 액션의 각 차원을 256개의 빈(bin)으로 이산화합니다. 훈련 데이터의 1분위수에서 99분위수까지 균일하게 나누어 이상치(outlier)의 영향을 줄입니다.
   - **토큰 매핑**: 이산화된 액션 ($0 \dots 255$)을 Llama 토크나이저의 가장 적게 사용되는 256개 토큰에 덮어씌워 매핑합니다.
   - **훈련 목표**: 표준 다음 토큰 예측(next-token prediction) 목표로, 예측된 액션 토큰에 대해서만 교차 엔트로피 손실(cross-entropy loss)을 평가합니다.
3. **훈련 데이터**:
   - **Open X-Embodiment 데이터셋 [1]**: 970k개의 실제 로봇 조작 궤적(trajectory)을 활용합니다. 다양한 로봇 구현체(embodiment), 장면, 작업이 포함됩니다.
   - **데이터 큐레이션**: 단일 팔 엔드 이펙터 제어, 최소한 하나의 3인칭 카메라가 있는 조작 데이터셋으로 제한합니다. Octo [5]의 데이터 혼합 가중치(mixture weights)를 사용하여 균형 잡힌 데이터셋을 구성합니다. DROID 데이터셋도 포함하려 했으나 학습 속도 저하로 후반에는 제외했습니다.
4. **주요 설계 결정**:
   - **VLM 백본**: Prismatic VLM이 IDEFICS-1 및 LLaVA보다 뛰어난 언어 기반 및 공간 추론 능력을 보여 채택했습니다.
   - **이미지 해상도**: $224 \times 224$ 픽셀을 사용했습니다. 더 높은 해상도($384 \times 384$ 픽셀)와 비교했을 때 성능 차이는 없었으나 훈련 시간이 3배 더 걸렸기 때문입니다.
   - **시각 인코더 미세 조정**: VLM 훈련 중 시각 인코더를 고정하는 것이 일반적으로 더 높은 성능을 보인다는 이전 연구와 달리, VLA에서는 시각 인코더를 미세 조정하는 것이 로봇 제어에 필요한 세밀한 공간 세부 정보를 포착하는 데 중요함을 발견했습니다.
   - **훈련 에포크**: 훈련 데이터셋을 27회 반복(epoch)하여 95% 이상의 액션 토큰 정확도를 달성했습니다. 이는 일반적인 LLM 또는 VLM 훈련보다 훨씬 많은 반복입니다.
   - **학습률**: $2 \times 10^{-5}$의 고정 학습률을 사용했습니다.
5. **훈련 및 추론 인프라**:
   - **훈련**: 64개의 A100 GPU 클러스터에서 14일 동안 (총 21,500 A100-시간) 배치 사이즈 2048로 훈련되었습니다.
   - **추론**: bfloat16 정밀도에서 15GB GPU 메모리를 사용하며, NVIDIA RTX 4090 GPU에서 약 6Hz로 실행됩니다. 4비트 양자화를 통해 메모리 사용량을 더욱 줄일 수 있습니다. 실시간 원격 액션 예측을 위한 원격 VLA 추론 서버를 제공합니다.

## 📊 Results

- **다중 로봇 플랫폼 직접 평가**:

  - WidowX (BridgeData V2) 및 Google 로봇 플랫폼에서 OpenVLA는 RT-2-X (55B)와 비슷한 성능을 보이거나, BridgeData V2에서는 16.5% 더 높은 절대 성공률을 달성했습니다 (OpenVLA는 7B).
  - RT-1-X 및 Octo에 비해 OpenVLA와 RT-2-X 모두 현저히 견고한 동작과 높은 성능을 보였습니다.
  - RT-2-X가 의미론적 일반화(semantic generalization)에서 약간 더 높았지만, OpenVLA는 다른 모든 작업 카테고리에서 동등하거나 더 나은 성능을 보였습니다.

- **새로운 로봇 설정에 대한 데이터 효율적인 적응**:

  - Franka-Tabletop 및 Franka-DROID 환경에서 10~150개의 데모만으로 OpenVLA를 미세 조정했습니다.
  - OpenVLA는 Octo 및 처음부터 훈련된 Diffusion Policy보다 높은 종합 성능을 달성했으며, 모든 테스트 작업에서 최소 50%의 성공률을 기록한 유일한 접근 방식이었습니다.
  - Diffusion Policy는 단일 지시 작업에서 강했지만, OpenVLA와 Octo는 언어 기반이 중요한 다중 지시 작업에서 더 나은 성능을 보였습니다.
  - OpenX 사전 훈련은 OpenVLA의 성능에 결정적인 역할을 했습니다 (OpenVLA (scratch) 결과에서 확인).

- **매개변수 효율적인 미세 조정 (PEFT)**:

  - LoRA(r=32 또는 r=64)는 전체 미세 조정(Full FT)과 동일한 성능을 달성하면서 훈련 매개변수를 1.4%로 줄이고 GPU 메모리 소비를 크게 절감했습니다.
  - LoRA를 사용하면 단일 A100 GPU에서 10-15시간 내에 OpenVLA를 미세 조정할 수 있어, 전체 미세 조정 대비 컴퓨팅 자원을 8배 절감합니다.

- **양자화를 통한 메모리 효율적인 추론**:
  - 4비트 양자화(int4)는 bfloat16 정밀도와 유사한 성능을 유지하면서 GPU 메모리 사용량을 절반 이상(16.8GB $\to$ 7.0GB) 줄였습니다.
  - 8비트 양자화(int8)는 추론 속도 저하로 인해 BridgeData V2에서 성능이 저하되었으나, 블로킹 제어(blocking control) 환경에서 추론 속도의 영향을 배제했을 때는 bfloat16 및 4비트 양자화와 유사한 성능을 보였습니다.

## 🧠 Insights & Discussion

OpenVLA는 범용 로봇 조작을 위한 최첨단 오픈 소스 VLA 모델이며, 즉시 사용 가능한 다중 로봇 제어 기능과 매개변수 효율적인 미세 조정을 통한 쉬운 적응성을 제공하여 VLA의 접근성을 높입니다. 융합 시각 인코더와 OpenX 데이터셋의 다양한 훈련이 성능 향상에 기여했습니다.

**제한 사항**:

- 현재 단일 이미지 관측만 지원합니다. 향후 다중 이미지 및 고유수용성(proprioceptive) 입력, 관측 이력 지원으로 확장이 필요합니다.
- ALOHA [90]와 같은 고주파수 제어(50Hz)를 위한 추론 처리량(inference throughput) 개선이 필요합니다. 액션 청킹(action chunking) 또는 투기적 디코딩(speculative decoding)과 같은 기술이 해결책이 될 수 있습니다.
- 현재 테스트 작업에서 90% 미만의 성공률을 보이며, 전반적인 신뢰성 향상의 여지가 있습니다.
- 컴퓨팅 한계로 인해 VLA 설계의 많은 질문(VLM의 크기, 로봇/인터넷 데이터 공동 훈련의 효과, 최적의 시각 특징 등)이 탐구되지 않은 상태입니다. OpenVLA의 공개를 통해 커뮤니티가 이러한 질문들을 공동으로 탐구하기를 희망합니다.
- 실세계 데이터로 사전 훈련된 모델(OpenVLA, Octo)이 시뮬레이션 환경에 미세 조정될 때 실세계 작업에 비해 성능 향상 폭이 작았는데, 이는 시뮬레이션과 실세계 환경 간의 도메인 격차(domain gap)를 시사합니다.

## 📌 TL;DR

오픈 소스 VLA (Vision-Language-Action) 모델인 OpenVLA는 로봇 조작 정책의 일반화 부족과 기존 VLA의 폐쇄성 문제를 해결합니다. Llama 2 (7B) 기반 VLM에 SigLIP과 DINOv2 융합 시각 인코더를 결합하고, 970k Open X-Embodiment 로봇 데모로 미세 조정되었습니다. OpenVLA는 7배 적은 매개변수로도 RT-2-X와 같은 폐쇄형 모델을 능가하는 최첨단 성능을 달성했으며, LoRA를 이용한 효율적인 미세 조정과 4비트 양자화를 통해 소비자용 GPU에서도 배포 및 적응이 가능합니다. 모든 코드와 모델이 오픈 소스로 공개되어 로봇 학습 연구를 가속화할 것으로 기대됩니다.
