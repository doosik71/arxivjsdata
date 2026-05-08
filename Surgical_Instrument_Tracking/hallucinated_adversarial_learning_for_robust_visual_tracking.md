# Hallucinated Adversarial Learning for Robust Visual Tracking

Qiangqiang Wu, Zhihui Chen, Lin Cheng, Yan Yan, Bo Li, Hanzi Wang (2019)

## 🧩 Problem to Solve

본 논문은 비주얼 트래킹(Visual Tracking) 분야에서 딥 컨볼루션 신경망(CNN)이 겪는 고질적인 문제인 **저데이터 환경에서의 과적합(Over-fitting)** 문제를 해결하고자 한다.

일반적인 CNN 기반 트래커는 대규모 학습 데이터에 크게 의존하지만, 온라인 트래킹 상황에서는 임의의 객체를 추적해야 하므로 충분한 학습 데이터를 수집하는 것이 불가능하다. 특히 온라인 학습 기반의 'Tracking-by-detection' 방식의 경우, 긍정 샘플(Positive samples)의 부족으로 인해 모델이 제한된 데이터에 과적합되어 추적 성능이 저하되는 문제가 발생한다. 따라서 본 연구의 목표는 인간의 상상력(Imagination) 또는 환각(Hallucination) 메커니즘을 모사하여, 적은 양의 데이터만으로도 다양하고 합리적인 긍정 샘플을 생성해 내는 강건한 트래커를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간이 과거의 경험을 통해 본 적 없는 대상의 다양한 모습(자세, 조명, 시점 등)을 상상할 수 있다는 점에 착안하여, 이를 신경망으로 구현한 **Adversarial Hallucinator (AH)**를 제안한 것이다.

주요 기여 사항은 다음과 같다.

1. **Adversarial Hallucinator (AH)**: 동일 정체성(Same-identity)을 가진 객체 쌍 사이의 비선형 변형(Non-linear deformations)을 학습하고, 이를 새로운 객체에 적용하여 다양한 긍정 샘플을 생성하는 GAN 기반의 구조를 제안하였다.
2. **Deformation Reconstruction (DR) Loss**: AH가 단순히 무작위 샘플을 생성하는 것이 아니라 실제 변형을 정확하게 학습하도록 돕는 자기지도 학습(Self-supervised) 방식의 손실 함수를 도입하였다.
3. **Selective Deformation Transfer (SDT)**: 타겟 객체와 시맨틱 특성이 유사한 데이터셋의 스니펫(Snippet)을 선택적으로 추출하여, 더 적절하고 합리적인 변형을 전이(Transfer)시키는 방법을 제안하였다.
4. **Hallucinated Adversarial Tracker (HAT)**: AH를 온라인 분류기(MDNet)와 결합하여 엔드투엔드(End-to-end) 방식으로 공동 최적화하는 트래킹 프레임워크를 구축하였다.

## 📎 Related Works

본 논문은 기존의 CNN 기반 트래킹 접근 방식을 크게 두 가지로 구분하여 설명한다.

- **One-stage Framework (예: Siamese networks)**: 대규모 데이터셋으로 오프라인 학습을 진행하여 일반화 성능을 높이지만, 온라인 적응성(Online adaptability)이 부족하여 정확도가 낮다는 한계가 있다.
- **Two-stage Framework (예: MDNet)**: 온라인 분류기를 업데이트하여 적응성을 높이지만, 온라인 샘플 부족으로 인한 과적합 문제에 취약하다.

또한, 기존의 데이터 증강(Data Augmentation) 방식들과의 차별점을 강조한다. UPDT, SINT++, VITAL과 같은 기존 연구들은 단순한 기하학적 변환(회전, 블러 등)이나 고정된 형태의 마스킹(Occlusion masks)을 사용하여 샘플을 생성하였다. 반면, 본 논문의 AH는 실제 데이터 쌍으로부터 **비선형 변형(Non-linear deformations)**을 직접 학습하여 전이하므로, 생성된 샘플이 실제 변형된 객체의 모습과 훨씬 더 가깝다는 차별점이 있다.

## 🛠️ Methodology

### 1. Adversarial Hallucinator (AH)

AH는 인코더($E_n$)와 디코더($D_e$)로 구성된 생성적 적대 신경망(GAN) 구조이다.

- **인코더($E_n$)**: 동일 정체성 객체 쌍 $(x_{a1}, x_{a2})$의 특징 벡터를 입력받아, 두 객체 사이의 변형을 나타내는 저차원 벡터 $z_a$를 추출한다.
- **디코더($D_e$)**: 추출된 변형 $z_a$와 새로운 객체 $x_{b1}$의 특징을 결합하여, 변형된 새로운 샘플 $\hat{x}_b$를 생성한다.
  $$\hat{x}_b = D_e([z_a, \phi(x_{b1})])$$
- **Adversarial Loss ($L_{adv}$)**: 판별기(Discriminator) $D$를 통해 생성된 샘플 $\hat{x}_b$가 실제 데이터 분포 $P_{data}$와 유사하도록 학습시킨다.
  $$L_{adv} = \min_{G} \max_{D} \mathbb{E}_{x_{b1}, x_{b2} \sim P_{data}} [\log D([\phi(x_{b1}), \phi(x_{b2})])] + \mathbb{E}_{x_{b1} \sim P_{data}, \hat{x}_b \sim P_{data}} [\log (1 - D([\phi(x_{b1}), \hat{x}_b]))]$$

### 2. Deformation Reconstruction (DR) Loss

단순히 $L_{adv}$만 사용하면 무작위로 정체성만 유지하는 샘플이 생성될 수 있다. 이를 방지하기 위해 본 논문은 **변형 재구성 손실(DR Loss)**을 제안한다.
만약 $\hat{x}_b$가 $x_{a1}$과 $x_{a2}$ 사이의 변형 $z_a$를 정확히 반영했다면, $\hat{x}_b$와 $x_{b1}$ 사이의 변형 $z_b$를 다시 $x_{a1}$에 적용했을 때 원래의 $x_{a2}$가 재구성되어야 한다는 원리를 이용한다.
$$L_{def} = ||D_e([z_b, \phi(x_{a1})]) - \phi(x_{a2})||^2$$
여기서 $z_b = E_n([\phi(x_{b1}), \hat{x}_b])$이다. 최종 손실 함수는 다음과 같다.
$$L_{overall} = L_{adv} + \lambda L_{def}$$

### 3. Selective Deformation Transfer (SDT)

모든 변형이 모든 객체에 적합한 것은 아니다(예: 사람의 포즈 변형을 자동차에 적용하는 것은 부적절함). 이를 해결하기 위해 타겟 객체 $x_e$와 시맨틱하게 유사한 스니펫을 소스 도메인에서 검색하여 사용한다.

- **스니펫 디스크립터**: ResNet34의 깊은 층에서 추출한 특징 벡터의 평균을 사용하여 각 스니펫의 대표 벡터 $\psi(s_i)$를 계산한다.
- **최근접 이웃 검색**: 타겟 객체 $\phi(x_e)$와 유클리드 거리가 가장 가까운 상위 $T$개의 스니펫을 선택하여 변형 전이에 사용한다.

### 4. Hallucinated Adversarial Tracker (HAT) 파이프라인

1. **Joint Model Initialization**: 초기 프레임에서 타겟 주변의 샘플과 AH를 통해 생성된 환각 긍정 샘플(Hallucinated positive samples)을 사용하여 AH와 분류기(MDNet)를 함께 학습시킨다.
2. **Online Detection**: 이전 프레임의 위치 주변에서 샘플을 추출하고 분류기를 통해 최적의 후보를 선택하며, Bounding box regression으로 위치를 정교화한다.
3. **Joint Model Update**: 매 프레임마다 새로운 긍정/부정 샘플과 AH 기반의 증강 샘플을 사용하여 AH와 분류기의 FC 레이어를 공동 업데이트한다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB-2013, OTB-2015, VOT-2016.
- **지표**: DPR (Distance Precision Rate), AUC (Area Under the Curve), EAO (Expected Average Overlap) 등.
- **비교 대상**: MDNet, VITAL, MetaSDNet, SiamRPN, CCOT 등 13개의 최신 트래커.

### 주요 결과

- **정량적 성능**:
  - **OTB-2013**: HAT는 **95.1%의 DPR**을 기록하며 1위 성능을 달성하였다. AUC 또한 69.6%로 베이스라인인 MDNet(66.8%)보다 2.8% 향상되었다.
  - **OTB-2015**: Accuracy 면에서 91.6%로 가장 높은 성능을 보였으며, AUC(66.9%) 또한 VITAL(67.0%)과 대등한 수준으로 최상위권에 위치하였다.
  - **VOT-2016**: EAO 0.32를 기록하며 MDNet(0.26) 대비 약 23%의 상대적 성능 향상을 보였으며, 특히 Failure(실패 횟수) 지표에서 16.52로 가장 안정적인 모습을 보였다.

- **Ablation Study**:
  - **SDT 효과**: SDT를 적용했을 때 DPR과 AUC가 모두 상승하였으며, 이는 시맨틱하게 유사한 변형을 사용하는 것이 더 합리적인 샘플 생성으로 이어진다는 것을 증명한다.
  - **샘플 비율**: 긍정 샘플과 부정 샘플의 비율 $r$을 $1/3$에서 $1/1$로 균형 있게 맞추었을 때 성능이 가장 크게 향상되었다.
  - **Online Update**: AH를 온라인에서 업데이트했을 때($HAT\text{-}w\text{-}Up$) 업데이트하지 않았을 때보다 성능이 유의미하게 높았으며, 이는 오프라인 학습 데이터와 온라인 데이터 사이의 도메인 갭(Domain gap)을 줄여주기 때문이다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 단순한 데이터 증강을 넘어 **'변형의 전이'**라는 개념을 도입하여, CNN 기반 트래커의 고질적인 문제인 데이터 부족 문제를 효과적으로 해결하였다. 특히 DR Loss를 통해 생성기의 무작위성을 제어하고, SDT를 통해 전이 가능한 변형만을 선택적으로 사용함으로써 생성된 샘플의 품질을 높였다. 또한 AH와 분류기를 공동 최적화함으로써, 분류기가 필요로 하는 '보완적인' 긍정 샘플을 AH가 생성하도록 유도한 점이 매우 효율적이다.

### 한계 및 논의사항

1. **추론 속도**: AH의 구조와 온라인 업데이트 과정이 추가됨에 따라 평균 트래킹 속도가 **1.6 FPS**로 매우 느리다. 실시간 성능이 중요한 애플리케이션에 적용하기 위해서는 연산 효율화가 필수적이다.
2. **특징 추출기 의존성**: 본 연구는 ResNet34 및 VGG-M의 사전 학습된 특징을 사용하였다. 특징 추출기의 성능이나 종류에 따라 AH가 학습하는 변형의 품질이 달라질 수 있다.
3. **데이터셋 편향**: ImageNet-VID라는 대규모 데이터셋에서 변형을 학습했으므로, 이 데이터셋에 포함되지 않은 완전히 새로운 형태의 변형이 발생하는 환경에서의 일반화 성능에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 인간의 상상력을 모방하여 부족한 온라인 학습 데이터를 보완하는 **Adversarial Hallucinator (AH)**와 이를 통합한 **HAT** 트래커를 제안한다. AH는 동일 객체 쌍의 비선형 변형을 학습해 새로운 객체에 적용하며, DR Loss와 SDT 방법을 통해 생성 샘플의 합리성을 보장한다. 실험 결과, OTB-2013에서 95.1%의 정밀도를 달성하는 등 SOTA 성능을 기록하였다. 이 연구는 비주얼 트래킹뿐만 아니라 Few-shot learning이나 Semi-supervised learning 등 데이터 부족 문제가 심각한 다양한 컴퓨터 비전 작업에 응용될 가능성이 높다.
