# Progressive Class-level Distillation

Jiayan Li, Jun Li, Zhourui Zhang, and Jianhua Xu (2025)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD)의 핵심 방법론 중 하나인 Logit Distillation(LD)에서 발생하는 정보 손실 문제를 해결하고자 한다. 기존의 LD 방식은 교사 모델(Teacher)과 학생 모델(Student)의 로짓(logits) 수준에서 정렬을 수행하는데, 이때 높은 신뢰도를 가진 클래스들이 증류 과정을 지배하게 된다. 이로 인해 낮은 확률을 가진 클래스들이 포함하고 있는 변별력 있는 정보(discriminating information)가 무시되는 경향이 있으며, 결과적으로 불충분하고 편향된 지식 전이가 일어나는 문제가 발생한다.

따라서 본 연구의 목표는 낮은 확률의 클래스들까지 효과적으로 활용하여, 학생 모델이 교사 모델의 지식을 보다 포괄적이고 세밀하게 학습할 수 있도록 하는 새로운 LD 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모든 클래스를 한꺼번에 증류하는 기존의 앙상블 방식에서 벗어나, 학습 난이도에 따라 단계적으로 지식을 전이하는 **Progressive Class-level Distillation (PCD)** 방식을 도입하는 것이다.

주요 설계 직관은 인간의 학습 과정과 유사하게, 먼저 지식을 단계적으로 축적하는 '점진적 학습(Progressive Learning)'을 수행하고, 이후 이를 다시 검증하고 정교화하는 '적응적 정제(Adaptive Refinement)' 과정을 거치는 양방향 단계별 증류(Bidirectional Stage-wise Distillation) 구조를 설계한 점이다. 이를 통해 어려운 클래스부터 쉬운 클래스까지 체계적으로 정렬함으로써 지식 전이의 효율성을 극대화한다.

## 📎 Related Works

논문은 지식 증류를 크게 Feature Distillation (FD)과 Logit Distillation (LD)의 두 그룹으로 분류하여 설명한다.

1.  **Feature Distillation (FD):** FitNet과 같이 교사 모델의 중간 레이어 특징을 모방하게 하여 로컬 패턴과 구조적 정보를 전이하는 방식이다. 이후 RKD, CRD, OFD 등 다양한 연구가 진행되었으나, 주로 중간 특징 맵의 정렬에 집중한다.
2.  **Logit Distillation (LD):** 교사 모델의 출력 확률 분포를 학생 모델이 모방하도록 하는 방식이다. 최근에는 다중 레벨 증류(Multi-level LD), 타겟/비타겟 클래스 분리(DKD), 커리큘럼 학습 기반의 온도 조절(CTKD) 등이 제안되었다.

기존 LD 방식들은 계산 효율성이 높고 적용이 쉽다는 장점이 있지만, 여전히 높은 확률의 클래스에 의존하여 낮은 확률 클래스의 중요성을 간과한다는 한계가 있다. PCD는 이러한 한계를 극복하기 위해 클래스 간의 예측 차이를 기반으로 우선순위를 정하고 단계별로 접근한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. 배경 및 기본 수식
기본적인 LD에서는 온도 계수 $\tau$를 적용한 Softmax 함수를 통해 교사와 학생의 확률 분포 $p$와 $q$를 계산한다.

$$p_i = \frac{\exp(z^t_i/\tau)}{\sum_{j=1}^C \exp(z^t_j/\tau)}, \quad q_i = \frac{\exp(z^s_i/\tau)}{\sum_{j=1}^C \exp(z^s_j/\tau)}$$

여기서 $z^t$와 $z^s$는 각각 교사와 학생의 로짓 출력이다. 일반적인 KD 손실 함수는 교차 엔트로피(CE) 손실과 KL 발산(KL divergence) 손실의 합으로 정의된다.

$$L^{KD} = \alpha \cdot L^{CE} + \beta \cdot \tau^2 \cdot KL(p_\tau, q_\tau)$$

### 2. Logit Difference Ranking (LDR)
PCD는 학습하기 어려운 클래스를 우선적으로 다루기 위해, 교사와 학생 모델 간의 로짓 차이를 계산하여 내림차순으로 정렬한다.

$$I = \text{argsort}(|z^t - z^s|, \downarrow)$$

여기서 $I$는 정렬된 클래스 인덱스의 시퀀스이며, 차이가 큰(즉, 학생 모델이 예측하기 어려워하는) 클래스가 우선순위를 갖게 된다.

### 3. Bidirectional Stage-wise Distillation (BSD)
정렬된 시퀀스를 바탕으로 증류 과정을 여러 단계($S$)로 나누어 수행하며, 두 가지 방향의 학습 프로세스를 거친다.

*   **Fine-to-Coarse Learning (F2CL):** 세밀한 부분에서 거친 부분으로 나아가는 과정으로, 지식을 점진적으로 축적한다.
*   **Coarse-to-Fine Learning (C2FL):** 반대로 거친 부분에서 세밀한 부분으로 돌아오며 지식을 정제한다.

각 단계 $i$에서 클래스 그룹의 크기 $m_i$는 다음과 같이 결정된다.
$$m^{F2CL}_i = \frac{C}{S-i+1}, \quad m^{C2FL}_i = \frac{C}{i} \quad (i=1, 2, \dots, S)$$

#### 세부 구현 절차:
1.  **Logit Masking:** 특정 단계의 그룹 $I_{i,j}$에 속하지 않는 클래스의 로짓은 $-\infty$로 마스킹하여 Softmax 계산 시 제외한다.
    $$z^s_{i,j} = \text{mask}(z^s, I_{i,j}), \quad z^t_{i,j} = \text{mask}(z^t, I_{i,j})$$
2.  **Weighted Distillation Mechanism (WDM):** 교사와 학생의 확률 분포 간 코사인 거리를 측정하여 가중치 $\lambda_{i,j}$를 부여함으로써, 학습이 부족한 클래스에 더 집중하게 한다.
    $$\lambda_{i,j} = 1 - \cos(p_{i,j}, q_{i,j})$$
    $$\text{가중치 적용 손실: } D_{i,j} = \lambda_{i,j} \cdot KL(p_{i,j} \| q_{i,j}) \cdot \tau^2$$
3.  **최종 손실 함수:**
    $$L = L^{CE} + \alpha \cdot (L^{F2CL}_{KL} + L^{C2FL}_{KL})$$

## 📊 Results

### 실험 설정
*   **데이터셋:** CIFAR, ImageNet, MS-COCO.
*   **평가 지표:** classification의 경우 Top-1, Top-5 accuracy, detection의 경우 Average Precision (AP).
*   **모델 구성:** VGG, ResNet, MobileNet, ShuffleNet 등 다양한 구조의 모델을 교사와 학생으로 설정하여 동종(Homogeneous) 및 이종(Heterogeneous) 환경에서 테스트하였다.

### 주요 결과
1.  **CIFAR 분류 성능:** vanilla KD 대비 비약적인 성능 향상을 보였다. 특히 ResNet32$\times$4 $\rightarrow$ ResNet8$\times$4 설정에서 Baseline 대비 Top-1 정확도를 3.01% 향상시켜 76.34%를 달성하였다. 또한, 이종 모델 간의 증류에서도 타 LD 방법론들보다 우수한 성능을 보였다.
2.  **ImageNet 분류 성능:** 대규모 데이터셋에서도 효과적임을 입증하였다. vanilla KD 대비 Top-1 정확도를 약 1.5%, Top-5 정확도를 0.9% 향상시켰으며, RC 및 LR 등의 최신 LD 방법보다 우월한 결과를 나타냈다.
3.  **MS-COCO 객체 검출 성능:** Faster-RCNN 기반의 검출 작업에서도 vanilla KD 대비 AP를 각각 0.31%, 0.14% 향상시키며 범용적인 적용 가능성을 보여주었다.

### 분석 및 소결
*   **Ablation Study:** LDR, F2CL, C2FL 세 가지 모듈을 모두 사용했을 때 최적의 성능이 나타났다. 특히 LDR을 제거했을 때 성능 저하가 뚜렷하여, '어려운 클래스 $\rightarrow$ 쉬운 클래스' 순의 정렬이 점진적 학습에 핵심적임을 확인하였다.
*   **하이퍼파라미터 분석:** 단계 수 $S$의 경우, 동종 모델은 $S=3$, 이종 모델은 $S=5$일 때 최적의 성능을 보였다. 이는 모델 간 구조적 차이가 클수록 더 많은 증류 단계가 필요함을 시사한다.

## 🧠 Insights & Discussion

본 논문은 기존 LD가 가진 '고확률 클래스 편향' 문제를 정확히 짚어내고, 이를 해결하기 위해 학습의 순서를 제어하는 전략을 제안하였다. 특히 단순히 순서를 정하는 것에 그치지 않고, 양방향(Fine-to-Coarse $\leftrightarrow$ Coarse-to-Fine) 학습 구조를 통해 지식의 축적과 정제를 동시에 꾀했다는 점이 인상적이다.

또한, 코사인 유사도 기반의 가중치 메커니즘(WDM)을 도입하여 모델이 스스로 부족한 부분을 인지하고 집중하게 만든 점은 학습의 효율성을 높이는 영리한 설계이다. t-SNE 시각화 결과에서도 PCD를 적용한 학생 모델의 표현력이 더 명확하게 구분되는 것이 확인되어, 제안 방법론이 실제로 클래스 간 변별력을 높였음을 알 수 있다.

다만, $S$ 값에 따라 성능이 달라지는데, 최적의 $S$를 찾기 위한 탐색 비용이 발생한다는 점과 이종 모델에서 더 많은 단계가 필요하다는 점은 실제 적용 시 고려해야 할 트레이드-오프(trade-off) 요소이다.

## 📌 TL;DR

PCD는 기존 Logit Distillation이 무시하던 저확률 클래스의 중요 정보를 회복하기 위해, **로짓 차이 기반의 순위 지정(LDR)**과 **양방향 단계별 증류(BSD)**를 제안한다. 어려운 클래스부터 점진적으로 학습하고 다시 정제하는 과정을 통해, CIFAR, ImageNet, MS-COCO 등 다양한 벤치마크에서 기존 KD 및 최신 LD 방법론들을 상회하는 성능을 입증하였다. 이 연구는 특히 모델 구조가 서로 다른 이종 모델 간의 지식 전이 효율을 높이는 데 기여할 가능성이 크다.