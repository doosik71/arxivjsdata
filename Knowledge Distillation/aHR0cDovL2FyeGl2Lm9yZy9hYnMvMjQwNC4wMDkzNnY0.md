# A Comprehensive Review of Knowledge Distillation in Computer Vision

Gousia Habib, Tausifa jan Saleem, Sheikh Musa Kaleem, Tufail Rouf, Brejesh Lall (2023/2024)

## 🧩 Problem to Solve

최근 딥러닝 기술, 특히 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)는 컴퓨터 비전 분야에서 혁신적인 성능 향상을 가져왔다. 그러나 이러한 모델들은 일반적으로 거대한 파라미터 수와 높은 연산 복잡도를 가지고 있어, 메모리와 전력이 제한된 리소스 제약 환경(resource-constrained environments)이나 실시간 응용 분야(real-time applications)에 배포하는 데 심각한 어려움이 있다.

본 논문의 목표는 복잡하고 거대한 모델의 지식을 작고 효율적인 모델로 압축하여 전달하는 기술인 Knowledge Distillation (KD, 지식 증류)의 최신 연구 동향을 종합적으로 분석하는 것이다. 특히 컴퓨터 비전의 다양한 작업(task)에서 KD가 어떻게 적용되고 있으며, 어떤 방법론들이 효율적인 모델 압축을 가능하게 하는지를 체계적으로 리뷰하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 컴퓨터 비전 분야에서의 Knowledge Distillation을 체계적으로 분류하고 분석한 종합 리뷰를 제공하는 것이다. 중심적인 설계 아이디어는 'Teacher-Student' 프레임워크로, 이미 잘 학습된 거대 모델(Teacher)이 가진 지식을 작은 모델(Student)이 모방하게 함으로써, Student 모델이 단독으로 학습했을 때보다 더 높은 성능을 내면서도 낮은 연산 비용을 유지하도록 하는 것이다.

이를 위해 논문은 KD의 지식 전송 유형, 증류 체계(schemes), 그리고 이미지 초해상도부터 멀티모달 모델에 이르기까지 광범위한 적용 사례를 다룬다.

## 📎 Related Works

논문에서는 모델의 복잡도를 줄이기 위한 기존의 접근 방식들을 다음과 같이 설명한다.

- **Network Pruning**: 중요도가 낮은 연결, 뉴런, 또는 필터를 제거하여 메모리 사용량과 추론 속도를 최적화한다.
- **Weight Quantization**: 32비트 부동 소수점 정밀도를 8비트 정수 등으로 낮추어 메모리 풋프린트를 줄이고 연산 속도를 높인다.
- **Weight Multiplexing/Sharing**: 여러 연결이 동일한 가중치 값을 공유하게 하여 파라미터 수를 줄인다.
- **Compact Network Design**: MobileNet이나 SqueezeNet과 같이 처음부터 적은 파라미터를 갖도록 설계된 효율적인 아키텍처를 사용한다.

기존 방식들이 주로 모델의 구조적 제거(Pruning)나 정밀도 저하(Quantization)에 집중하는 반면, Knowledge Distillation은 Teacher 모델이 데이터에 대해 학습한 '암시적 지식(implicit knowledge)' 또는 'Soft targets'를 Student에게 전달함으로써 성능 저하를 최소화하며 압축한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

Knowledge Distillation은 기본적으로 **Teacher-Student Framework**를 따른다. 거대 모델인 Teacher가 입력 데이터에 대해 생성한 예측값이나 중간 특징 맵(feature maps)을 가이드라인으로 삼아, 작은 모델인 Student가 이를 모방하도록 학습시킨다.

### 주요 지식 전송 유형

1. **Response-based Knowledge Transfer**: Teacher의 최종 출력층에서 나오는 확률 분포(Soft targets)를 Student가 예측하도록 학습한다.
2. **Feature-based Knowledge Transfer**: Teacher의 중간 레이어에서 추출된 특징 표현(feature representations)을 Student가 모방하게 한다.
3. **Relation-based Knowledge Transfer**: 데이터 샘플 간의 관계(예: 샘플 A와 B의 유사도)를 Teacher가 어떻게 파악했는지를 Student에게 전달한다.

### 학습 목표 및 손실 함수

Student 모델은 일반적으로 실제 정답(Ground-truth)과의 차이를 줄이는 손실과 Teacher의 예측과 일치시키려는 증류 손실의 가중 합을 최소화하는 방향으로 학습된다.

$$L_{distill} = \alpha \cdot T^2 \cdot \text{cross\_entropy}(y_{student}, y_{teacher}/T) + (1 - \alpha) \cdot \text{cross\_entropy}(y_{student}, t)$$

여기서 각 변수의 의미는 다음과 같다.

- $y_{student}, y_{teacher}$: 각각 Student와 Teacher 모델의 출력(logits)이다.
- $t$: 실제 정답 레이블(true labels)이다.
- $T$: Temperature(온도) 파라미터로, Softmax 함수를 통과시키기 전 logit을 나눠줌으로써 확률 분포를 더 부드럽게(soften) 만들어 'Dark Knowledge'(정답 외 클래스 간의 관계)를 더 잘 학습하게 한다.
- $\alpha$: 두 손실 항 사이의 가중치를 조절하는 하이퍼파라미터이다.

### 증류 체계 (Distillation Schemes)

- **Offline Distillation**: 미리 학습된 고정된 Teacher 모델로부터 Student가 지식을 배운다.
- **Online Distillation**: Teacher와 Student가 동시에 학습하며 서로 상호작용한다.
- **Self-distillation**: 모델 스스로가 자신의 더 큰 버전이나 이전 상태의 지식을 전수한다.

## 📊 Results

본 논문은 리뷰 논문으로서 다양한 선행 연구의 정량적 결과를 요약하여 제시한다.

- **Image Super-Resolution**: Data-Free KD 방식을 통해 추가 학습 데이터 없이도 Set5 데이터셋에서 33.06dB의 PSNR을 달성하였다.
- **Face Recognition**: Exclusivity Consistency Regularized KD (ECRKD)를 적용하여 LFW 데이터셋에서 0.97%라는 낮은 에러율을 기록하였다.
- **Person Search**: Expert-guided KD를 통해 CUHK-SYSU 데이터셋에서 mAP 84.1%를 달성하였으며, Teacher 모델 대비 연산 시간을 75% 단축하였다.
- **Semantic Segmentation**: Structured KD를 통해 PASCAL VOC 데이터셋에서 83.6%의 mIoU를 기록하며 SOTA 성능을 보였다.
- **Multimodal Models**: Tiny CLIP의 경우, CLIP ViT-B/32 모델의 크기를 절반으로 줄이면서도 성능 손실을 최소화하고 학습 속도를 크게 향상시켰다.

## 🧠 Insights & Discussion

### 강점 및 유효성

KD는 단순히 모델을 작게 만드는 것을 넘어, Teacher 모델이 가진 일반화 능력을 Student에게 전이시킴으로써 Student 모델 단독 학습 시보다 더 높은 정확도를 얻게 할 수 있다. 특히 'Soft targets'를 활용함으로써 클래스 간의 상관관계를 학습하는 것이 모델의 강건성(robustness) 향상에 기여한다는 점이 확인되었다.

### 한계 및 미해결 질문

논문은 효율적인 KD 설계를 위해 다음과 같은 변수들이 매우 중요함을 언급한다.

- **Teacher-Student Gap**: Teacher와 Student 사이의 용량(capacity) 차이가 너무 크면 오히려 학습 효율이 떨어질 수 있다.
- **Loss Function Selection**: 작업(Task)에 따라 단순한 Logit 일치가 아닌 Feature나 Relation 기반의 손실 함수 선택이 필수적이다.
- **Computational Cost**: 일부 최신 KD 방법론은 학습 과정에서 Teacher의 연산량이 추가되어 학습 시간이 매우 길어지는 문제가 있다.

### 비판적 해석

본 논문은 광범위한 분야를 다루고 있어 전반적인 흐름을 파악하기에 매우 훌륭하지만, 각 방법론의 상세한 하이퍼파라미터 튜닝 방법이나 구체적인 구현 디테일보다는 결과 중심의 요약에 치중되어 있다. 또한, 최근의 LLM 기반 Vision-Language Model로의 패러다임 변화에 따른 KD의 역할 변화에 대해 더 깊은 논의가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 컴퓨터 비전 모델의 거대화로 인한 배포 문제를 해결하기 위해, Teacher 모델의 지식을 Student 모델로 전이하는 **Knowledge Distillation (KD)** 기술을 종합적으로 분석한 리뷰 보고서이다. Response, Feature, Relation 기반의 전송 방식과 Offline, Online, Self-distillation 체계를 체계화하였으며, 이미지 분류부터 세그멘테이션, 멀티모달 모델(CLIP)까지의 적용 사례와 성능 향상 결과를 제시하였다. 이 연구는 향후 리소스 제약 환경에서 고성능 비전 모델을 구현하려는 연구자들에게 필수적인 가이드라인과 향후 연구 방향(도메인 적응, 해석 가능한 증류 등)을 제시한다는 점에서 중요한 가치가 있다.
