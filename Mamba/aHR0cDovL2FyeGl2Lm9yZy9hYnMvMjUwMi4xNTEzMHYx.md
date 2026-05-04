# TransMamba: Fast Universal Architecture Adaption from Transformers to Mamba

Xiuwei Chen, Sihao Lin, Xiao Dong, Zisheng Chen, Meng Cao, Jianhua Han, Hang Xu, Xiaodan Liang (2025)

## 🧩 Problem to Solve

본 논문은 현대 딥러닝의 주류인 Transformer 아키텍처의 연산 효율성 문제와 최신 sub-quadratic 아키텍처인 Mamba의 학습 비용 문제를 동시에 해결하고자 한다.

Transformer는 유연한 확장성 덕분에 유니모달 및 멀티모달 파운데이션 모델에서 널리 사용되고 있으나, Attention 메커니즘의 계산 복잡도가 입력 길이의 제곱($\mathcal{O}(N^2)$)에 비례한다는 치명적인 단점이 있다. 이는 메모리 사용량과 연산 비용을 급격히 증가시켜 모델의 최적화와 확장을 어렵게 만든다.

반면, Mamba와 같은 State Space Model(SSM) 기반 아키텍처는 선형 복잡도($\mathcal{O}(N)$)를 가지면서도 전역적 인식(global awareness) 능력을 갖추고 있다. 하지만 이러한 sub-quadratic 모델을 특정 태스크를 위해 처음부터(from scratch) 학습시키는 것은 막대한 계산 자원과 시간이 소요되며, 이는 높은 탄소 배출과 같은 환경적 비용을 초래한다.

따라서 본 연구의 목표는 이미 대량의 데이터로 학습된 기존 Transformer 모델의 지식을 Mamba 아키텍처로 효율적으로 전이(transfer)하여, 적은 비용으로 고성능의 Mamba 모델을 구축하는 **TransMamba** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 강력한 표현 능력을 Mamba의 효율적인 구조로 이식하는 '교차 아키텍처 학습(cross-architecture learning)'에 있다. 주요 기여 사항은 다음과 같다.

1.  **빠르고 보편적인 전이 학습 프레임워크**: 사전 학습된 Transformer 모델의 지식을 새로운 SSM 기반 모델로 전이하는 2단계 전략을 통해 학습 효율성과 최종 성능을 동시에 향상시킨다.
2.  **WSAB (Weight Subcloning and Adaptive Bidirectional distillation)**:
    *   **Weight Subcloning**: 구조적/차원적 차이를 극복하여 Transformer의 가중치를 Mamba에 직접 초기화하는 기법을 제안한다.
    *   **Adaptive Bidirectional Distillation**: 레이어 간 유사도에 따라 가중치를 다르게 부여하는 적응형 증류 방식과, 이미지 데이터의 특성을 반영한 순방향/역방향 양방향 증류 방식을 통해 최적화를 정밀하게 수행한다.
3.  **Cross-Mamba 모듈**: Mamba 아키텍처에 부족한 멀티모달 상호작용 능력을 부여하기 위해, 언어 인식을 시각적 특징에 통합하는 Cross-Mamba 모듈을 제안하여 비전-언어 태스크 성능을 높였다.

## 📎 Related Works

### 기존 연구 및 한계
*   **Transformers**: ViT, CLIP, LLaVA 등 다양한 모델이 성공을 거두었으나, 앞서 언급한 quadratic complexity로 인해 고해상도 이미지나 긴 시퀀스 처리 시 연산 부담이 매우 크다.
*   **State Space Models (SSMs)**: S4, Mamba 등이 등장하며 선형 복잡도로 긴 시퀀스를 처리할 수 있게 되었다. 특히 Mamba는 선택적 스캔 메커니즘(S6)을 통해 데이터 의존적인 처리가 가능하다. 하지만 비전 분야에서의 적용은 아직 초기 단계이며, 학습 비용이 높다.
*   **Transfer Learning**: Transformer의 지식을 CNN으로 전이하는 연구나, NLP 분야에서 Transformer를 Mamba로 증류하는 연구가 일부 존재했다. 그러나 비전 및 멀티모달 분야에서 Mamba로의 지식 전이는 이미지 정보의 복잡성으로 인해 연구가 매우 부족한 상태였다.

### 차별점
TransMamba는 단순한 가중치 복사가 아니라, **특징 정렬(Feature Calibration)**, **적응형 양방향 증류**, 그리고 **가중치 서브클로닝**을 결합하여 구조가 완전히 다른 두 아키텍처 간의 간극을 메운다. 특히 멀티모달 상호작용을 위한 전용 모듈(Cross-Mamba)을 도입했다는 점이 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. Preliminary: SSM 및 Mamba
SSM은 연속적인 시스템을 통해 입력 $x(t)$를 출력 $y(t)$로 변환하며, 기본적으로 다음과 같은 상미분 방정식(ODE)을 따른다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$
Mamba는 이를 이산화(discretization)하여 처리하며, 특히 **Selective Scan Mechanism (S6)**을 통해 입력 데이터 $x$에 따라 파라미터 $\bar{A}, \bar{B}, C, \Delta$가 결정되도록 하여 유연한 정보 선택이 가능하게 한다.

### 2. TransMamba 전체 파이프라인

#### (1) Feature Calibration (특징 정렬)
Transformer와 Mamba는 잠재 공간(latent space)의 차원이 다를 수 있다. 이를 해결하기 위해 Mamba의 차원을 Transformer에 맞게 zero-padding하고, 단순한 MLP 레이어를 사용하여 두 모델의 특징을 동일한 정렬 공간으로 투영한다.

#### (2) Adaptive Bidirectional Distillation (적응형 양방향 증류)
단순한 로짓(logits) 증류는 중간 레이어의 불일치를 해결하지 못한다. 본 논문은 코사인 유사도 기반의 지식 증류를 수행한다.

*   **적응형 손실 함수**: 모든 레이어를 동일하게 맞추는 대신, 유사도가 낮은 레이어에 더 높은 가중치를 부여하여 균형 잡힌 최적화를 수행한다.
$$L_{AdaptDistill} = \sum_{i=1}^{N} \sum_{j=1}^{N} \frac{\cos\theta_j}{\cos\theta_i} (1 - \cos\theta_i)$$
*   **양방향 증류 (Bidirectional Distillation)**: 이미지 처리를 위해 Mamba는 보통 순방향과 역방향 스캔을 모두 수행한다. 이때 Transformer의 정적인 특징을 그대로 사용하면 과적합 또는 과소적합 문제가 발생한다. 따라서 순방향 과정에서는 Transformer 특징을 그대로 사용하고, 역방향 과정에서는 Transformer 특징을 **반전(Reverse)**시켜 정렬함으로써 정밀하게 학습한다.
*   **최종 손실 함수**:
$$L_{total} = \alpha L_{task} + (1-\alpha)(L_{forward} + L_{backward})$$

#### (3) Weight Subcloning (가중치 서브클로닝)
구조적 차이(Attention vs SSM)와 차원 차이를 극복하기 위한 가중치 초기화 전략이다.
*   **구조적 대응**: SSM 부분을 제외한 나머지 부분(MLP, Layer Norm 등)은 Transformer의 파라미터로 초기화한다.
*   **차원 대응**: 모든 가중치를 가져오는 대신, 사전 학습 후 미세 조정(fine-tuning) 시 변화량이 적은 '중요한' 뉴런/가중치를 선택하여 Mamba의 차원에 맞게 이식한다.

#### (4) Cross-Mamba Module (멀티모달 상호작용)
Mamba는 기본적으로 Transformer의 Cross-Attention과 같은 상호작용 기법이 부족하다. 이를 해결하기 위해 Mamba의 연산 구조 $Y = C(MX)$에서 $Q, K, V$ 역할을 다음과 같이 정의하여 모달리티 간 상호작용을 가능케 한다.
*   $Q = S^C(x)$, $K = S^B(x)$, $V = x$ 로 설정하고, $\Delta$ 또한 유사한 모달리티 입력을 사용하도록 하여 텍스트와 이미지 간의 상호작용을 촉진한다.

## 📊 Results

### 실험 설정
*   **데이터셋**:
    *   유니모달: CIFAR100, ImageNet-100, ImageNet-1000
    *   멀티모달: LLaVA-1.5 (VQA), MSR-VTT, DiDeMo (비디오 검색)
*   **비교 모델**: VMamba, PlainMamba, VisionMamba(ViM), DeiT, LLaVA-3.2 등
*   **지표**: Accuracy (Top-1), Recall@k, Mean Rank

### 주요 결과
1.  **이미지 분류 (Image Classification)**:
    *   TransMamba는 Vanilla Mamba 모델들보다 일관되게 높은 성능을 보인다. 예를 들어, TransPMamba-T는 PMamba-T 대비 정확도가 최대 2.65%p 향상되었다.
    *   학습 곡선 분석 결과, TransMamba는 훨씬 빠른 수렴 속도를 보이며 더 낮은 Loss에 도달한다.
2.  **시각적 질의응답 (VQA)**:
    *   Trans-LLaVA (0.6B 파라미터)는 더 큰 모델인 LLaVA-3.2-1B보다 GQA, VQA, VisWiz 등의 벤치마크에서 더 우수한 성능을 보였으며, 3B 모델의 성능에 근접하는 결과를 냈다.
3.  **비디오 검색 (Video Retrieval)**:
    *   MSR-VTT 데이터셋에서 R@1 지표 기준, VideoMamba(40.9)보다 TransVideoMamba(41.6)가 0.7%p 높은 성능을 기록했다.
4.  **효율성 분석 (Ablation Study)**:
    *   **데이터 양**: 데이터의 50%~75%만 사용해도 전체 데이터를 사용한 경우와 거의 동일한 성능에 도달한다. 이는 Transformer의 지식 전이가 학습 시간을 획기적으로 단축시켰음을 의미한다.
    *   **증류 전략**: 단순 Feature/Logit 증류보다 본 논문의 WSAB 방식이 압도적으로 높은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 아키텍처가 완전히 다른 Transformer와 Mamba 사이의 지식 전이가 가능하다는 것을 실험적으로 입증했다. 특히 단순한 증류를 넘어 가중치 서브클로닝(Weight Subcloning)과 적응형 양방향 증류를 결합함으로써, Mamba 모델이 처음부터 학습될 때 겪는 불안정성과 막대한 비용 문제를 해결했다. 또한, Mamba의 구조적 한계인 멀티모달 상호작용 부족을 Cross-Mamba 모듈로 보완하여 범용성을 확보했다.

### 한계 및 논의사항
*   **초기화 의존성**: 대형 모델 학습 시 SSM 파라미터를 NLP 도메인에서 사전 학습된 Mamba 모델로 초기화했을 때 안정성이 높아졌다는 점은, 여전히 SSM 자체의 초기화 문제가 까다롭다는 것을 시사한다.
*   **가정**: 본 연구는 기존의 강력한 Transformer 모델이 존재한다는 가정하에 작동한다. 따라서 전이할 '교사 모델'의 성능이 학생 모델의 상한선을 결정짓는 구조적 한계가 있다.
*   **비판적 시각**: 데이터 사용량을 75% 이하로 줄여도 성능이 유지된다는 결과는 고무적이나, 이는 특정 데이터셋에 국한된 결과일 수 있으며, 매우 거대한 데이터셋에서도 동일한 효율이 나타날지는 추가 검증이 필요하다.

## 📌 TL;DR

**TransMamba**는 사전 학습된 Transformer의 지식을 효율적으로 Mamba 아키텍처로 전이하는 프레임워크이다. **가중치 서브클로닝(Weight Subcloning)**과 **적응형 양방향 증류(WSAB)**를 통해 학습 시간과 데이터 사용량을 대폭 줄이면서도(데이터 75% 미만 사용), 이미지 분류, VQA, 비디오 검색 등 다양한 태스크에서 기존 Mamba 모델보다 뛰어난 성능을 달성했다. 특히 **Cross-Mamba** 모듈을 통해 SSM 기반 모델의 멀티모달 상호작용 능력을 강화하였으며, 이는 향후 저비용·고효율의 차세대 비전-언어 모델 구축에 중요한 역할을 할 것으로 기대된다.