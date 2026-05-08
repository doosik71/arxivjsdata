# Domain Adaptive Knowledge Distillation for Driving Scene Semantic Segmentation

Divya Kothandaraman, Athira Nambiar, Anurag Mittal (2020)

## 🧩 Problem to Solve

자율주행 시스템의 인지 단계에서 필수적인 Semantic Segmentation은 실무 적용 시 두 가지 핵심적인 난제에 직면한다. 첫째는 **메모리 제약(Memory Constraints)** 문제이다. 높은 성능을 내는 최신 모델들은 대부분 파라미터 수가 많고 깊은 구조를 가지고 있어, 차량 내 탑재되는 모바일 기기와 같은 제한된 메모리 환경에서 실시간으로 구동하기 어렵다. 둘째는 **도메인 갭(Domain Gap)** 문제이다. 특정 도메인(Source)에서 학습된 모델은 조명, 날씨, 계절 등의 변화로 인해 다른 도메인(Target)에서 성능이 급격히 저하되는 경향이 있다. 특히 타겟 도메인에 대한 정답지(Ground Truth) 데이터를 수집하고 레이블링하는 작업은 비용이 매우 많이 든다.

본 논문의 목표는 이 두 가지 문제를 동시에 해결하는 것이다. 즉, 계산 비용이 큰 Teacher 네트워크로부터 도메인 적응 지식을 효율적으로 전수받아, 메모리 제약이 적은 경량 Student 네트워크가 타겟 도메인에서도 높은 성능을 낼 수 있도록 하는 'Domain Adaptive Knowledge Distillation' 프레임워크를 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Unsupervised Domain Adaptation(UDA)**과 **Knowledge Distillation(KD)**을 하나의 통합된 프레임워크로 결합하여, 모델 압축과 도메인 적응을 동시에 달성하는 것이다. 이를 위한 주요 기여 사항은 다음과 같다.

1. **Multi-level Distillation (MLD) 전략**: 지식 전수를 단순히 출력층에 한정하지 않고, Feature space와 Output space라는 두 가지 레벨에서 동시에 수행하여 전수되는 지식의 양과 질을 높였다.
2. **Pseudo-teacher Label 기반의 새로운 손실 함수**: Teacher 네트워크의 Soft prediction을 통해 생성한 Pseudo label을 활용하여, 타겟 도메인에서 정답지가 없는 상황을 극복하고 Student의 학습을 가이드하는 Cross Entropy 손실 함수를 도입하였다.
3. **네 가지 증류 패러다임(Distillation Paradigms) 제시**: Source 전용, Target 전용, Source+Target 동시, 그리고 단계적 전수(c $\to$ d)라는 네 가지 시나리오를 통해 최적의 학습 경로를 분석하였다.

## 📎 Related Works

### 관련 연구 및 한계

- **Domain Adaptation (DA)**: Adversarial training, Reconstruction, Distribution distance minimization 등을 통해 Source와 Target 간의 특징 공간을 정렬하여 도메인 갭을 줄이려는 시도가 많았다. 하지만 이러한 연구들은 주로 모델의 크기나 효율성보다는 정확도 향상에만 집중하였다.
- **Knowledge Distillation (KD)**: 거대 모델(Teacher)의 'Dark Knowledge'(클래스 간 확률 분포)를 작은 모델(Student)에게 전달하여 성능을 보존하며 모델을 압축하는 기법이다. 주로 이미지 분류나 객체 검출에 적용되었으며, Semantic Segmentation에서도 일부 시도가 있었으나 도메인 적응 문제와 결합된 경우는 드물었다.

### 기존 접근 방식과의 차별점

기존의 일부 연구들이 KD를 DA의 수단으로 사용하거나, Semi-supervised DA 환경에서 KD를 적용한 적은 있으나, 본 논문은 완전히 레이블이 없는 **Unsupervised Domain Adaptation (UDA)** 환경에서 자율주행 씬 세그멘테이션을 위해 모델 압축과 도메인 적응을 동시에 설계한 최초의 시도 중 하나이다. 또한, 단순한 지식 전수를 넘어 Pseudo label을 통해 타겟 도메인의 Ground Truth 대용치(Proxy)를 제공함으로써 UDA 성능을 극대화하였다.

## 🛠️ Methodology

### 전체 시스템 구조

전체 파이프라인은 무거운 Teacher 네트워크($T$)와 경량 Student 네트워크($S$)로 구성된다. 두 네트워크는 먼저 Multi-level DA 전략을 통해 사전 학습되며, 이후 Teacher의 가중치는 동결(Frozen)된 상태에서 Student에게 지식을 전수한다.

### 훈련 목표 및 손실 함수

#### 1. Baseline Domain Adaptation (DA)

Teacher와 Student 모두 기본적으로 다음과 같은 DA 손실 함수를 통해 사전 학습된다.

- **Segmentation Loss ($\mathcal{L}_{seg}$)**: Source 도메인 이미지에 대해 적용되는 Cross Entropy 손실이다.
$$\mathcal{L}_{seg} = -\sum_{h,w} \sum_{c \in C} Y_{s(h,w,c)} \log(P_{s(h,w,c)})$$
- **Adversarial Loss ($\mathcal{L}_{adv}$)**: Target 도메인의 특징을 Source와 유사하게 만들기 위해 Discriminator를 속이는 손실이다.
$$\mathcal{L}_{adv} = -\sum_{h,w} \log(D(P_{t})_{h,w,1})$$

#### 2. Multi-level Distillation (MLD) 손실 함수

Student는 다음 세 가지 손실 함수의 조합을 통해 Teacher로부터 지식을 학습한다.

- **KL Divergence Loss ($\mathcal{L}_{KL}$)**: Teacher와 Student의 출력 확률 분포를 유사하게 만든다. Output 레벨($\mathcal{L}_{KL,out}$)과 Feature 레벨($\mathcal{L}_{KL,feat}$) 모두에서 적용된다. (Feature 레벨에서는 특징 맵을 확률 맵으로 투영하여 계산한다.)
$$\mathcal{L}_{KL} = \lambda_{KL} * \sum_{i} KL(q_{s_{i}} || q_{t_{i}})$$

- **MSE Loss ($\mathcal{L}_{MSE}$)**: Feature space에서 두 네트워크의 특징 맵이 물리적으로 유사해지도록 'Hard alignment'를 수행한다.
$$\mathcal{L}_{MSE} = \lambda_{MSE} * \sum_{i} ||(q_{s_{i}} - q_{t_{i}})||^{2}$$

- **Pseudo-teacher Label Loss ($\mathcal{L}_{pseudoT}$)**: Teacher의 예측값 중 가장 확률이 높은 클래스를 Pseudo label($Y_{pseudoT}$)로 생성하여 Student가 이를 학습하게 한다. 이는 특히 타겟 도메인에서 Ground Truth의 역할을 대신한다.
$$\mathcal{L}_{pseudoT} = \lambda_{pseudoT} * \sum_{h,w} \sum_{c} Y_{pseudoT} \log(P_{student})$$

### 최종 목적 함수 (Overall Objective)

전체 지식 증류 손실 $\mathcal{L}_{distill}$은 Source($s$)와 Target($t$) 도메인에 대한 손실의 합이며, 타겟 도메인에는 스케일링 인자 $\lambda_{target}$을 곱하여 가중치를 조절한다.
$$\mathcal{L}_{distill} = [\mathcal{L}_{s-KL} + \mathcal{L}_{s-MSE} + \mathcal{L}_{s-pseudoT}] + \lambda_{target} * [\mathcal{L}_{t-KL} + \mathcal{L}_{t-MSE} + \mathcal{L}_{t-pseudoT}]$$

### 증류 패러다임 (Distillation Paradigms)

1. **(a) Source Distillation**: Source 이미지로만 증류 수행.
2. **(b) Target Distillation**: Target 이미지로만 증류 수행.
3. **(c) Source + Target Distillation**: 두 도메인 모두에서 동시에 증류 수행.
4. **(d) Sequential Target Distillation**: (c) 방식으로 초기화한 후, Target 도메인 증류로 파인튜닝 수행.

## 📊 Results

### 실험 설정

- **데이터셋**: Cityscapes (CS), Berkeley Deep Drive (BDD), GTA5.
- **시나리오**: Real-to-Real (BDD $\to$ CS), Synthetic-to-Real (GTA5 $\to$ CS).
- **지표**: mIoU (mean Intersection over Union), Pixel Accuracy.
- **모델**: Teacher (DRN-D-38, 26.5M params) / Student (DRN-D-22, 16.4M params).

### 주요 결과

1. **성능 향상**: 모든 시나리오에서 지식 증류를 적용한 Student가 적용하지 않은 Student보다 월등한 성능을 보였다.
2. **패러다임 비교**: 패러다임 (d)가 가장 높은 성능을 기록하였다. BDD $\to$ CS 케이스에서 mIoU 44.15를 달성하여, 단순 Student(38.3)뿐만 아니라 Teacher(42.33)보다도 더 높은 성능을 보였다.
3. **SOTA 비교**: 동일한 백본(DRN-D-22)을 사용한 AdaptSegNet과 비교했을 때, 본 제안 방법이 mIoU 기준 약 4~6포인트 더 높은 성능을 보였으며, 모델 크기는 더 작았다.
4. **Ablation Study**: 세 가지 손실 함수($\mathcal{L}_{KL}, \mathcal{L}_{MSE}, \mathcal{L}_{pseudoT}$)를 모두 사용했을 때 최적의 결과(mIoU 42.33)가 나왔으며, 특히 Pseudo label 기반의 $\mathcal{L}_{pseudoT}$가 성능 향상에 가장 크게 기여함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 가장 놀라운 점은 **경량 Student 모델이 더 무거운 Teacher 모델의 성능을 추월**했다는 점이다. 이는 단순히 지식을 복제한 것이 아니라, KD 과정이 일종의 정규화(Regularization) 역할을 하여 Student 모델이 타겟 도메인에 더 잘 일반화되도록 도왔기 때문으로 해석된다. 특히 Pseudo label이 타겟 도메인의 가이드라인 역할을 수행하며 UDA의 고질적인 문제인 '레이블 부재'를 효과적으로 해결하였다.

### 한계 및 비판적 해석

- **Teacher 의존성**: Student의 성능이 Teacher의 성능에 크게 의존한다. 만약 Teacher가 타겟 도메인에서 완전히 잘못된 예측을 한다면, Pseudo label이 오히려 노이즈로 작용하여 Student의 성능을 저하시킬 위험이 있다.
- **계산 복잡도**: 학습 단계에서는 Teacher와 Student 두 모델을 모두 유지하고 복잡한 손실 함수를 계산해야 하므로 학습 시간이 증가한다. 다만, 추론(Inference) 단계에서는 Student만 사용하므로 실용적이다.
- **하이퍼파라미터 민감도**: $\lambda_{KL}, \lambda_{MSE}, \lambda_{pseudoT}$ 등 조절해야 할 하이퍼파라미터가 많아, 최적의 조합을 찾는 데 많은 실험적 비용이 소모될 수 있다.

## 📌 TL;DR

본 논문은 자율주행 환경에서 **메모리 제약**과 **도메인 갭**이라는 두 가지 난제를 동시에 해결하기 위해 **Domain Adaptive Knowledge Distillation (DAKD)** 프레임워크를 제안한다. Feature와 Output 레벨에서 동시에 지식을 전수하는 MLD 전략과 Teacher의 Pseudo label을 활용한 손실 함수를 통해, 경량 Student 모델이 타겟 도메인에서 Teacher 모델보다도 높은 성능을 낼 수 있음을 입증하였다. 이 연구는 자율주행용 실시간 세그멘테이션 모델을 효율적으로 구축하는 데 중요한 방법론을 제시한다.
