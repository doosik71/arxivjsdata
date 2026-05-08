# Sleep Stage Classification using Multimodal Embedding Fusion from EOG and PSM

Olivier Papillon, Rafik Goubran, James Green, Julien Larivière-Chartier, Caitlin Higginson, Frank Knoefel, Rebecca Robillard (2025)

## 🧩 Problem to Solve

본 논문은 수면 장애 진단을 위한 수면 단계 분류(Sleep Stage Classification)의 효율성과 접근성을 높이는 것을 목표로 한다. 현재 수면 단계 분류의 골드 표준(Gold Standard)은 뇌파(Electroencephalography, EEG) 기반의 수면다원검사(Polysomnography, PSG)이다. 그러나 EEG 센서는 피부 준비 과정이 복잡하고, 숙련된 임상의가 필요하며, 전문적인 실험실 인프라가 요구된다는 단점이 있어 가정 기반의 원격 수면 모니터링에 적용하기 어렵다.

따라서 본 연구는 EEG보다 훨씬 덜 침습적이고 설치가 간편한 안구전도(Electrooculography, EOG)와 압력 감지 매트(Pressure-Sensitive Mats, PSM)를 활용하여, 5단계의 수면-각성 상태를 정확하게 분류하는 시스템을 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 서로 다른 성격의 데이터인 PSM과 EOG 신호를 통합하기 위해 멀티모달 임베딩 모델인 **ImageBind**를 최초로 도입했다는 점이다. 연구진은 PSM 데이터를 비디오(Video) 형태로, EOG 신호를 오디오(Audio) 형태로 간주하여 ImageBind의 공유 임베딩 공간(Shared Embedding Space)에 정렬시켰다. 이를 통해 의료 데이터라는 제한된 라벨링 데이터 환경에서도 사전 학습된(Pre-trained) 모델의 지식을 활용하여 높은 분류 정확도를 달성할 수 있음을 입증하였다.

## 📎 Related Works

### 1. 기존 연구의 접근 방식 및 한계

- **PSM 기반 연구**: 기존에는 압력 매트 데이터를 사용하여 수면 자세를 분류하거나, 단순한 수면-각성 여부만을 판별하는 수준(TCN 네트워크 활용 등)에 머물렀다.
- **EOG 기반 연구**: EOGNet이나 SE-Resnet-Transformer와 같은 딥러닝 모델이 제안되었으나, 주로 단일 모달리티(Single-modality)에 의존하여 정보의 부족함이 있었다.
- **멀티모달 접근**: EOG와 EEG를 결합한 연구가 있었으나, 여전히 침습적인 EEG가 포함되어 원격 모니터링의 한계를 완전히 해결하지 못했다.
- **비디오/오디오 모델**: ViViT(Video Vision Transformer)나 MBT(Multimodal Bottleneck Transformer)와 같은 최신 모델들이 존재하지만, 이를 수면 단계 분류라는 의료 도메인에 적용하여 검증한 사례는 부족했다.

### 2. 본 연구의 차별점

본 연구는 EEG를 완전히 배제하고, 비침습적인 PSM과 EOG만을 결합한다. 특히, 의료 전용 모델이 아닌 일반 도메인에서 학습된 ImageBind 모델을 전이 학습(Transfer Learning) 시켜 의료 데이터의 도메인 갭을 극복하고 성능을 향상시켰다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

시스템은 PSM과 EOG 데이터를 입력받아 ImageBind 임베딩을 추출하고, 이를 결합하여 최종적으로 5개의 수면 단계(Wake, NREM1, NREM2, NREM3, REM)를 분류하는 구조이다.

### 2. 데이터 전처리 및 구성 요소

- **EOG 데이터**: 좌우 양측 채널의 신호를 수집하여 0에서 1 사이로 정규화한다. ImageBind 입력을 위해 이를 듀얼 채널 WAV 오디오 파일로 저장하고, $16\text{kHz}$로 업샘플링한 후 $128\text{-mel}$ 스펙트로그램(Spectrogram)으로 변환한다.
- **PSM 데이터**: $18 \times 8$ 해상도의 저해상도 압력 이미지 시퀀스를 생성한다. 이는 시간 흐름에 따른 이미지의 연속이므로 비디오 데이터로 처리된다.
- **데이터 분할**: 30초 단위의 에포크(Epoch)로 데이터를 분할하며, 총 85명의 환자 데이터(약 63,236 에포크)를 사용한다.

### 3. 학습 절차 및 방정식 설명

- **임베딩 추출**: PSM 비디오와 EOG 오디오는 각각 ImageBind의 전용 인코더를 통해 동일한 차원의 벡터로 변환된다.
- **특징 융합(Fusion)**: 각 모달리티에서 추출된 임베딩 벡터 $v_{\text{PSM}} \in \mathbb{R}^{1024}$와 $v_{\text{EOG}} \in \mathbb{R}^{1024}$를 단순 연결(Concatenation)하여 하나의 통합 벡터를 생성한다.
  $$v_{\text{fused}} = [v_{\text{PSM}}; v_{\text{EOG}}] \in \mathbb{R}^{2048}$$
- **분류 단계**: 통합된 $2048 \times 1$ 벡터는 선형 레이어(Linear Layer)를 통과하여 5개 클래스에 대한 확률 값으로 매핑된다.
- **학습 전략**:
  - **Linear Probing**: ImageBind의 가중치를 고정한 채 마지막 분류 레이어만 학습시킨다.
  - **Fine-tuning**: 전체 모델의 가중치를 미세 조정하여 도메인 적응력을 높인다.
- **하이퍼파라미터**: AdamW 옵티마이저, 초기 학습률 $1.4 \times 10^{-7}$, 가중치 감쇠(Weight Decay) $0.005$, 6 에포크 동안 학습을 수행하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 실제 수면 클리닉에서 수집된 85명의 환자 데이터.
- **평가 지표**: 5-way 분류 정확도(Accuracy) 및 매크로 평균 F1 스코어(Macro-averaged F1 score).
- **검증 방법**: 5-폴드 교차 검증(5-fold Cross-validation)을 사용하여 환자 간 데이터 누수를 방지하였다.

### 2. 정량적 결과

실험 결과, ImageBind를 활용한 멀티모달 융합 방식이 가장 우수한 성능을 보였다.

| 방법론 | 사용 모달리티 | 정확도 (Accuracy) | F1 (Macro) |
| :--- | :--- | :---: | :---: |
| ViViT | PSM video | $0.399$ | $0.164$ |
| MBT | PSM video & Single-channel EOG | $0.631$ | $0.543$ |
| DeepSleepNet | Single-channel EOG | $0.743$ | $0.682$ |
| **ImageBind (Fine-tuned)** | **PSM video & Dual-channel EOG** | $\mathbf{0.745}$ | $\mathbf{0.683}$ |

### 3. 주요 분석

- **모달리티의 영향**: PSM 단독 모델(ViViT)은 성능이 매우 낮았으나, EOG를 추가했을 때 성능이 비약적으로 상승하였다. 이는 수면 단계, 특히 REM과 NREM을 구분하는 데 EOG 신호가 결정적인 역할을 하기 때문이다.
- **융합의 이점**: EOG 단독 모델인 DeepSleepNet($0.743$)보다 PSM을 결합한 ImageBind($0.745$)가 더 높은 성능을 보였다. 이는 신체 움직임을 포착하는 PSM 데이터가 보완적인 정보를 제공함을 시사한다.
- **도메인 적응력**: Fine-tuning을 하지 않은 Linear Probing 상태에서도 정확도 $0.690$을 기록하여, 일반 도메인에서 학습된 ImageBind가 의료 데이터에서도 강건한 기초 표현력을 가짐을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과

본 연구는 의료 분야의 고질적인 문제인 '라벨링된 데이터의 부족' 문제를 사전 학습된 거대 멀티모달 모델(ImageBind)의 전이 학습을 통해 효과적으로 해결하였다. 특히 RGB 비디오와 일반 오디오로 학습된 모델이 저해상도 압력 이미지와 생체 전기 신호(EOG)라는 전혀 다른 도메인에서도 작동한다는 점은 매우 고무적이다.

### 2. 한계 및 분석

혼동 행렬(Confusion Matrix) 분석 결과, REM, Wake, NREM2 단계는 비교적 잘 구분하였으나, NREM1, NREM2, NREM3 사이의 구분에는 어려움이 있었다. 이는 수면 단계의 전이가 점진적으로 일어나기 때문에, EEG 없이 EOG와 PSM만으로는 인접 단계 간의 미세한 차이를 식별하는 데 한계가 있음을 의미한다.

### 3. 비판적 해석

정확도 측면에서 DeepSleepNet(단일 EOG)과 ImageBind(멀티모달)의 차이가 매우 근소하다($0.743$ vs $0.745$). 이는 PSM의 추가 기여도가 예상보다 낮을 수 있음을 시사하며, 향후 연구에서는 PSM의 해상도를 높이거나 더 정교한 융합 메커니즘을 도입하여 PSM의 잠재력을 완전히 끌어낼 필요가 있다.

## 📌 TL;DR

본 논문은 침습적인 EEG 대신 **EOG(안구전도)와 PSM(압력 매트)을 결합하여 수면 단계를 분류하는 멀티모달 딥러닝 프레임워크**를 제안하였다. 특히 일반 도메인의 멀티모달 임베딩 모델인 **ImageBind**를 전이 학습시켜, 데이터가 부족한 의료 환경에서도 높은 분류 성능(정확도 $0.745$)을 달성하였다. 이 연구는 향후 병원 밖 가정 환경에서 환자의 수면 상태를 비침습적으로 정밀하게 모니터링할 수 있는 기술적 토대를 마련하였다는 점에서 큰 의미가 있다.
