# MammoFL: Mammographic Breast Density Estimation using Federated Learning

Ramya Muthukrishnan, Angelina Heyler, Keshava Katti, Sarthak Pati, Walter Mankowski, Aprupa Alahari, Michael Sanborn, Emily F. Conant, Christopher Scott, Stacey Winham, Celine Vachon, Pratik Chaudhari, Despina Kontos, Spyridon Bakas

## 🧩 Problem to Solve

본 연구는 유방촬영술(Mammography) 이미지에서 유방 밀도(Breast Density)를 정량적으로 추정하는 과정을 자동화하고자 한다. 유방 밀도는 유방암의 주요 위험 인자일 뿐만 아니라, 밀도가 높은 조직은 암 조직을 가리는 '마스킹 효과(masking effect)'를 일으켜 유방촬영술의 민감도를 저하시키는 원인이 된다.

현재 유방 밀도 측정은 영상의학 전문의의 시각적 등급 판정(BI-RADS)이나 일부 상용 소프트웨어(Quantra, Volpara 등)에 의존하고 있다. 하지만 상용 소프트웨어는 비용이 많이 들고 내부 작동 방식의 해석력이 낮으며, 특정 메타데이터에 의존하여 부정확한 결과를 낼 가능성이 있다. 또한, 기존의 연구용 도구들은 단일 기관의 소규모 데이터셋으로 학습되어 새로운 데이터에 대한 일반화 성능(Generalization)이 떨어진다는 한계가 있다. 의료 데이터의 특성상 여러 기관의 데이터를 한곳에 모으는 것은 개인정보 보호 및 소유권 문제로 인해 매우 어렵다. 따라서 본 논문은 환자의 프라이버시를 보호하면서도 여러 기관의 데이터를 활용해 모델의 일반화 성능을 높일 수 있는 Federated Learning(연합 학습) 기반의 유방 밀도 추정 시스템을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 유방 밀도 추정을 위한 엔드-투-엔드(End-to-End) CNN 파이프라인인 **MammoFL**을 제안하고, 이를 연합 학습(Federated Learning) 환경에서 구현하여 그 효용성을 입증한 것이다.

중심적인 설계 아이디어는 다음과 같다.

1. **이단계 세그멘테이션 구조**: 전체 이미지에서 유방 영역을 먼저 찾고, 그 내부에서 다시 치밀 조직(Dense tissue) 영역을 분리하는 두 개의 U-Net 구조를 설계하여 정량적인 유방 밀도(Percent Density, PD)를 계산한다.
2. **프라이버시 보존형 다기관 학습**: 데이터를 중앙 서버로 전송하지 않고, 각 기관(Collaborator)에서 로컬 학습을 수행한 뒤 모델의 가중치(Weights)만을 집계(Aggregation)하는 연합 학습 방식을 적용하여 데이터 유출 없이 다기관 데이터의 이점을 취한다.

## 📎 Related Works

기존의 유방 밀도 측정 방식은 다음과 같이 분류된다.

- **BI-RADS**: 전문의가 시각적으로 등급을 매기는 방식으로, 정량적인 면적이나 부피를 측정하기보다 암의 마스킹 가능성을 등급화하는 것에 집중한다.
- **상용 소프트웨어(Quantra, Volpara)**: X-ray 빔 상호작용 모델을 통해 부피 기반의 밀도를 추정하지만, 메타데이터 의존성이 높고 공간적 맵을 제공하지 않아 해석력이 부족하다.
- **연구용 도구(LIBRA 등)**: 일부 도구는 공개되어 있으나, 대부분 단일 기관의 작은 데이터셋으로 학습되어 일반화 능력이 부족하다.

본 연구의 MammoFL은 전통적인 방식(예: Deep-LIBRA)이 치밀 조직 세그멘테이션에 전통적인 방법론을 섞어 쓰는 것과 달리, 전체 과정을 CNN 기반의 파이프라인으로 구성하여 자동화 수준을 높였으며, 특히 연합 학습을 통해 다기관 데이터 부족 문제를 해결하려 했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

MammoFL은 두 개의 U-Net 모델로 구성된 순차적 파이프라인이다.

- **첫 번째 U-Net (Breast Segmentation)**: 전처리된 유방촬영술 이미지에서 배경과 흉근(Pectoralis muscle)을 제외한 유방 영역만을 세그멘테이션한다.
- **두 번째 U-Net (Dense Tissue Segmentation)**: 첫 번째 모델이 생성한 유방 마스크를 통해 배경을 제거한 이미지를 입력으로 받아, 유방 내부에서 치밀 조직(Dense tissue) 영역만을 분리한다.

최종적인 유방 밀도($PD$)는 다음과 같은 수식으로 계산된다.
$$\text{PD} = \frac{\text{Estimated Dense Tissue Area}}{\text{Estimated Breast Area}}$$

### 2. 모델 아키텍처 및 학습 절차

- **아키텍처**: 두 모델 모두 $\text{ResNet34}$를 백본(Backbone)으로 사용하는 U-Net 구조를 채택하였다.
- **전처리**: DICOM 이미지에서 메타데이터 태그를 제거하고, $512 \times 512$ 크기로 리사이징하며, $\text{min-max scaling}$을 통해 픽셀 강도를 $[0, 1]$ 범위로 정규화한다.
- **학습 설정**: $\text{Adam optimizer}$를 사용하였으며, 학습률(Learning rate)은 $1\text{e-4}$, Weight decay는 $1\text{e-4}$, 배치 크기는 $16$으로 설정하여 30 에포크(Epoch) 동안 학습하였다. 일반화 성능 향상을 위해 랜덤 플리핑(Flipping)과 같은 공간적 변환 증강(Data augmentation)을 적용하였다.

### 3. 연합 학습(Federated Learning) 절차

본 연구는 $\text{OpenFL}$ 라이브러리를 사용하여 Aggregator-Server 프레임워크를 구현하였다.

- **로컬 학습**: 각 기관(UPHS, MC)의 'Collaborator' 머신에서 로컬 데이터를 이용해 네트워크 가중치를 업데이트한다.
- **가중치 집계**: 'Aggregator' 머신은 각 기관에서 전송된 업데이트된 가중치들을 수집하여 가중 평균(Weighted average)을 계산한다. 이때 가중치는 각 기관이 보유한 데이터셋의 크기에 비례하여 결정된다.
- **데이터 보안**: 실제 환자 데이터는 각 기관 내부에 머물며, 네트워크 가중치만을 공유하므로 기관 간 데이터 공유 없이 다기관 학습 효과를 얻을 수 있다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Mayo Clinic(MC)과 University of Pennsylvania Health System(UPHS)의 유방촬영술 이미지.
- **평가 지표**:
  - **MAE (Mean Absolute Error)**: 실제 PD 값과 추정 값 사이의 평균 절대 오차.
  - **Spearman's correlation ($\rho$)**: 실제 값과 추정 값 사이의 상관관계.
  - **DSC (Dice-Sorensen Coefficient)**: 세그멘테이션의 정확도를 측정하는 지표.
      $$\text{DSC} = \frac{2|\text{GT} \cap \text{Pred}|}{|\text{GT}| + |\text{Pred}|}$$
- **비교 대상(Baselines)**:
  - 단일 기관 학습 모델 ($\text{Centralized MC}$, $\text{Centralized UPHS}$)
  - 데이터 통합 중앙 집중 학습 모델 ($\text{Centralized UPHS + MC}$)

### 2. 주요 결과

- **일반화 성능**: 단일 기관에서 학습된 모델은 타 기관 데이터에 대해 매우 낮은 성능을 보였다. 반면, $\text{MammoFL}$과 중앙 집중식 다기관 학습 모델은 두 기관 모두에서 높은 일반화 성능을 나타냈다.
- **FL vs Centralized**: $\text{MammoFL}$의 성능은 중앙 집중식 다기관 학습 모델보다는 통계적으로 유의미하게 낮았으나($p < 0.05$), 단일 기관 모델들보다는 월등히 뛰어났다. 이는 연합 학습이 프라이버시를 보호하면서도 중앙 집중식 학습에 근접한 성능을 낼 수 있음을 시사한다.
- **Gold-Standard 비교**: LIBRA로 생성한 라벨이 아닌, 실제 정답지인 Cumulus PD 라벨과 비교했을 때, $\text{MammoFL}$의 상관계수는 $\rho=0.7703$으로 나타났다. 이는 학습에 사용된 LIBRA 라벨 자체의 상관관계($\rho=0.7012$)보다 높은 수치이며, CNN이 노이즈가 섞인 라벨에서도 강건한 패턴을 찾아내어 더 정확한 예측을 수행했음을 보여준다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과

본 연구는 유방 밀도 추정이라는 구체적인 의료 영상 작업에서 연합 학습이 매우 효과적인 유즈케이스(Use-case)가 될 수 있음을 입증하였다. 특히 다기관 데이터 학습이 단순히 데이터 양을 늘리는 것을 넘어, 모델이 보지 못한 새로운 데이터(Unseen data)에 대한 일반화 능력을 확보하는 데 필수적임을 정량적으로 보여주었다.

### 2. 한계 및 비판적 해석

- **라벨의 신뢰성**: 학습에 사용된 ground-truth가 수동으로 작성된 gold-standard가 아니라 LIBRA 알고리즘에 의해 생성된 합성 라벨(Synthetic labels)이라는 점이 한계이다. 저자들은 이를 보완하기 위해 Cumulus 라벨과 비교 분석을 수행하였으나, 근본적으로는 실제 수동 라벨링 데이터로 학습했을 때 더 높은 성능 향상이 있을 것으로 판단된다.
- **성능 격차(Privacy-Utility Trade-off)**: 연합 학습 모델이 중앙 집중식 학습 모델보다 성능이 낮게 나타난 점은 연합 학습의 전형적인 한계인 '프라이버시 보호와 모델 성능 간의 트레이드오프'를 보여준다. 더 정교한 집계 알고리즘이나 최적화 기법이 필요함을 시사한다.
- **데이터 범위**: 표준 뷰(Standard views) 이미지들만 사용하였으며, 가슴 크기가 커서 여러 장의 겹친 이미지(Overlapping views)가 필요한 사례는 처리하지 못했다.

## 📌 TL;DR

본 논문은 유방 밀도를 정량적으로 측정하기 위해 두 개의 U-Net을 활용한 엔드-투-엔드 파이프라인인 **MammoFL**을 제안하고, 이를 **연합 학습(Federated Learning)**으로 학습시켜 다기관 데이터 활용 시의 프라이버시 문제와 일반화 성능 문제를 동시에 해결하고자 하였다. 실험 결과, 연합 학습은 단일 기관 학습보다 훨씬 뛰어난 일반화 능력을 보였으며, 중앙 집중식 다기관 학습에 근접한 성능을 달성하였다. 이 연구는 의료 분야에서 데이터 공유 없이도 협력 학습을 통해 진단 도구의 성능을 높일 수 있는 실질적인 방법론을 제시하였다는 점에서 향후 다기관 의료 AI 연구에 중요한 기반이 될 것으로 보인다.
