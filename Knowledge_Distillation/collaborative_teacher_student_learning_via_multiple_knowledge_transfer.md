# Collaborative Teacher-Student Learning via Multiple Knowledge Transfer

Liyuan Sun, Jianping Gou, Baosheng Yu, Lan Du, Dacheng Tao (2021)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD) 과정에서 발생하는 정보의 제한성과 모델 간의 역량 차이(Capacity Gap) 문제를 해결하고자 한다. 기존의 KD 방법론들은 주로 개별 인스턴스의 특징(Instance features)이나 인스턴스 간의 관계(Instance relations) 중 한 가지 유형의 지식만을 특정 전략을 통해 전달하는 경향이 있었다.

특히, 널리 사용되는 오프라인 증류(Offline Distillation) 방식은 미리 학습된 거대 교사 모델과 작은 학생 모델의 구조가 고정되어 있어 학습 역량의 한계가 있으며, 교사 모델을 사전 학습시키기 위해 대규모 데이터셋이 필요하다는 단점이 있다. 온라인 증류(Online Distillation)와 자기 증류(Self-distillation)가 이러한 한계를 극복하기 위한 대안으로 제시되었으나, 이들 역시 단일 지식 소스(주로 개별 인스턴스)에 의존하며, 온라인 방식의 경우 피어 네트워크(Peer networks) 간의 출력값 차이로 인해 인스턴스 일관성(Instance consistency)이 떨어지는 문제가 발생한다.

따라서 본 연구의 목표는 다양한 유형의 지식을 서로 다른 증류 전략을 통해 통합적으로 전달할 수 있는 unified framework를 구축하여, 모델의 일반화 성능을 높이고 압축 효율을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 자기 학습(Self-learning)과 협력 학습(Collaborative learning)을 결합한 **CTSL-MKT (Collaborative Teacher-Student Learning via Multiple Knowledge Transfer)** 프레임워크를 제안하는 것이다.

이 프레임워크의 중심 직관은 각 피어 네트워크가 자기 증류를 통해 스스로를 강화하는 동시에, 온라인 증류를 통해 상대 네트워크와 상호 보완적인 지식을 교환하게 하는 것이다. 이때 전달되는 지식을 단순히 출력값(Response-based)에 한정하지 않고, 데이터 간의 구조적 관계(Relation-based)까지 확장하여 다중 지식 전송(Multiple Knowledge Transfer)을 구현함으로써 학습의 풍부함을 더했다.

## 📎 Related Works

본 논문에서는 기존의 지식 증류 접근 방식을 세 가지 관점에서 분석하며 차별점을 제시한다.

1. **Self-Distillation**: 동일한 네트워크 내에서 지식을 전달하여 오버피팅을 줄이고 일반화 능력을 높이는 방법이다. 하지만 모델 스스로의 응답 기반 지식에만 의존한다는 한계가 있다.
2. **Collaborative Learning (Online Distillation)**: 교사와 학생이 동시에 학습하며 상호 작용하는 방식(예: Deep Mutual Learning, DML)이다. 이는 고정된 교사 모델의 제약을 없애주지만, 개별 인스턴스의 응답 기반 지식만을 교환하므로 전송되는 지식의 양이 제한적이다.
3. **Structural Knowledge (Relation-based KD)**: 개별 샘플의 출력이 아닌, 샘플 간의 상대적 관계(예: 거리, 각도)를 보존하는 방식(예: RKD)이다. 이는 구조적 정보를 제공하지만, 개별 샘플이 가진 직접적인 지식을 간과하는 경향이 있다.

CTSL-MKT는 이러한 개별 접근 방식들의 한계를 극복하기 위해 응답 기반 지식과 관계 기반 지식을 모두 사용하며, 이를 자기 증류와 온라인 증류라는 두 가지 전략으로 통합하여 상호 보완적으로 작동하게 설계되었다.

## 🛠️ Methodology

CTSL-MKT는 두 개의 피어 네트워크 $N_1, N_2$가 상호 학습하고 동시에 자기 학습을 수행하는 구조이다. 전체 학습 프로세스는 크게 두 단계로 나뉘며, 최종 목적 함수는 세 가지 손실 함수의 가중 합으로 정의된다.

### 1. 교사-학생 상호 학습 (Teacher-Student Mutual Learning)

상호 학습은 두 네트워크 간에 응답 기반 지식과 관계 기반 지식을 양방향으로 전달하는 과정이다.

**응답 기반 지식 전송 (Response-Based Knowledge Transfer)**
개별 인스턴스의 출력값(Soft logits)을 일치시키기 위해 KL Divergence 손실 함수를 사용한다. 네트워크 $N_k$가 $N_{k'}$로부터 학습할 때의 손실은 다음과 같다.
$$L_{KL}(p_k, p_{k'}) = \sum_{x \in X} \sum_{i=1}^{m} \sigma_i(z_{k'}(x), 1) \log \frac{\sigma_i(z_{k'}(x), 1)}{\sigma_i(z_k(x), 1)}$$
여기서 $\sigma$는 소프트맥스 함수이며, $z$는 로짓(logits) 값을 의미한다.

**관계 기반 지식 전송 (Relation-Based Knowledge Transfer)**
데이터 간의 구조적 관계를 보존하기 위해 거리 기반(Distance-wise)과 각도 기반(Angle-wise) 함수를 결합하여 사용한다.

- **거리 기반 함수**: 두 샘플 $x_u, x_v$ 사이의 유클리드 거리를 정규화하여 유사도를 측정한다.
- **각도 기반 함수**: 세 샘플 $x_u, x_v, x_w$ 사이의 각도(Cosine similarity)를 통해 구조적 관계를 측정한다.
두 네트워크 간의 관계 차이를 최소화하기 위해 Huber loss $R(\cdot)$를 적용한 관계 증류 손실 $L_{RD}$를 정의한다.
$$L_{RD} = L_{DD} + \beta_1 L_{AD}$$
최종적인 상호 증류 손실 $L^{MD}_k$는 다음과 같이 정의된다.
$$L^{MD}_k = L_{RD} + \beta_2 L_{KL}(p_k, p_{k'})$$

### 2. 학생 자기 학습 (Student Self-learning)

피어 네트워크 간의 출력값이 너무 다를 경우 상호 학습의 효율이 떨어질 수 있다. 이를 보완하기 위해 사전 학습된 자신의 출력값 $\bar{p}^t_k$를 타겟으로 삼아 스스로를 지도하는 자기 증류 손실 $L^{SD}_k$를 도입한다.
$$L^{SD}_k(p^t_k, \bar{p}^t_k) = \sum_{x \in X} \sum_{i=1}^{m} \sigma^t_i(\bar{z}_k(x), t) \log \frac{\sigma^t_i(\bar{z}_k(x), t)}{\sigma^t_i(z_k(x), t)}$$

### 3. 전체 학습 절차 및 최종 손실 함수

학습은 (1) 각 네트워크를 개별적으로 사전 학습시키는 단계와 (2) 정의된 통합 손실 함수를 통해 협력 학습을 수행하는 단계로 진행된다. 각 네트워크 $N_k$의 최종 목적 함수는 다음과 같다.
$$L^{KD}_k = \alpha L^{CE}_k + \beta L^{MD}_k + \gamma L^{SD}_k$$
여기서 $L^{CE}_k$는 정답 라벨과의 교차 엔트로피 손실이며, $\alpha, \beta, \gamma$는 각 손실의 기여도를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-10, CIFAR-100, Tiny-ImageNet, Market-1501 (Re-ID 태스크).
- **비교 대상**: DML (응답 기반 온라인), Tf-KD (자기 증류), RKD (관계 기반 온라인).
- **네트워크 구조**: ResNet, MobileNetV2, ShuffleNetV2, VGG, AlexNet, SqueezeNet 등 다양한 조합(동일 구조 및 서로 다른 구조)을 사용하였다.

### 주요 결과

1. **분류 성능 향상**: CIFAR-10 및 CIFAR-100 데이터셋에서 CTSL-MKT는 모든 비교 대상보다 높은 Top-1 정확도를 기록하였다. 특히 MobileNetV2와 ShuffleNetV2 같은 경량 네트워크 조합에서 baseline 대비 뚜렷한 성능 향상을 보였다.
2. **강건성 확인**: Tiny-ImageNet 실험 결과, 모델 크기가 작아 지식 전송 효율이 떨어지는 상황에서도 CTSL-MKT는 다중 지식 전송을 통해 안정적으로 성능을 끌어올렸다.
3. **Re-ID 성능**: Market-1501 데이터셋에서 mAP 지표 기준 DML(+1.22%), RKD(+0.8%), Tf-KD(+4.69%)보다 우수한 성능을 보였다.
4. **수렴 속도**: 학습 곡선 분석 결과, CTSL-MKT는 자기 학습(SL) 덕분에 DML이나 RKD보다 더 빠르게 수렴하며, Tf-KD와 유사한 수렴 속도를 보이면서도 최종 성능은 더 높게 나타났다.

### 절제 연구 (Ablation Study)

- **결과**: 모든 전략(MLI, MLR, SLI)을 모두 사용한 Full Model(Case A)이 가장 높은 성능을 보였다.
- **자기 증류의 중요성**: 자기 증류(SLI)를 제외한 경우(Case B) 성능 하락이 가장 컸다. 이는 자기 학습이 피어 네트워크 간의 출력 다양성으로 인한 불안정성을 상쇄하고 학습 효율을 높이는 결정적인 역할을 함을 시사한다.

## 🧠 Insights & Discussion

본 논문은 지식 증류에서 **'어떤 지식을(What)', '어떤 전략으로(How)'** 전달하느냐가 성능에 결정적인 영향을 미친다는 것을 실험적으로 증명하였다.

가장 주목할 점은 자기 학습과 협력 학습의 결합이다. 기존의 온라인 협력 학습(DML 등)은 두 모델의 출력이 너무 다를 경우 서로 잘못된 방향으로 가이드할 위험이 있는데, CTSL-MKT는 자기 증류를 통해 각 모델의 기본 성능을 견고히 다짐으로써 이러한 '다양성 문제(Diversity issue)'를 해결하였다. 또한, 응답 기반의 '개별 일관성'과 관계 기반의 '구조적 일관성'을 동시에 추구함으로써 더 풍부한 표현 학습이 가능함을 보여주었다.

다만, 다중 지식 전송을 위해 $\alpha, \beta, \gamma$ 및 $\beta_1, \beta_2$ 등 다수의 하이퍼파라미터를 튜닝해야 한다는 점이 실용적인 관점에서의 부담이 될 수 있다. 또한, 관계 기반 지식을 추출하기 위해 2-tuple 및 3-tuple 샘플 조합을 생성해야 하므로 계산 복잡도가 증가할 가능성이 있다.

## 📌 TL;DR

본 논문은 응답 기반 지식과 관계 기반 지식을 동시에 전달하며, 자기 증류와 온라인 협력 학습을 통합한 **CTSL-MKT** 프레임워크를 제안한다. 실험을 통해 단일 지식/전략 기반의 KD보다 다중 지식 전송이 모델의 일반화 성능을 유의미하게 향상시킴을 입증하였으며, 특히 자기 학습이 협력 학습의 불안정성을 보완한다는 점을 밝혔다. 이 연구는 향후 더 복잡한 지식 유형을 통합하는 모델 압축 및 전송 연구에 중요한 기초를 제공할 것으로 보인다.
