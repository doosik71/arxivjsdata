# 1st Place Solution to NeurIPS 2022 Challenge on Visual Domain Adaptation

Daehan Kim, Minseok Seo, YoungJin Jeon, Dong-Geol Choi (2022)

## 🧩 Problem to Solve

본 논문은 산업 폐기물 분류를 위한 시맨틱 세그멘테이션(Semantic Segmentation) 작업에서 Unsupervised Domain Adaptation(UDA) 문제를 해결하고자 한다. 딥러닝 모델은 일반적으로 학습 데이터의 분포(Source domain)와 테스트 데이터의 분포(Target domain)가 다를 때 성능이 급격히 저하되는 도메인 시프트(Domain Shift) 현상을 겪는다.

특히 산업 폐기물 분류와 같은 특수 도메인에서는 레이블링 된 데이터(Labeled data)를 확보하는 비용이 매우 높기 때문에, 레이블이 없는 타겟 도메인 데이터를 활용하여 모델의 일반화 성능을 높이는 UDA 기술이 필수적이다. 본 연구의 목표는 VisDA 2022 챌린지에서 제시된 폐기물 데이터셋에 대해 최적의 도메인 적응 성능을 보이는 $\text{SIA\_Adapt}$ 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 강력한 사전 학습된 표현(Transferable representation)을 확보하고, 이를 기반으로 점진적인 자기 학습(Self-training)과 가중치 앙상블(Model Soup)을 적용하는 것이다.

1. **강력한 사전 지식의 활용**: 기존의 SOTA 모델들이 사용하는 MiT(Mix Transformer) 백본 대신, 더 거대한 데이터셋인 ImageNet-22K로 사전 학습된 $\text{ConvNeXt-L}$ 백본을 채택하여 초기 도메인 전이 능력을 극대화하였다.
2. **정제된 Pseudo-label 기반의 Self-training**: 초기 적응 모델을 통해 생성된 의사 레이블(Pseudo-labels) 중 신뢰도가 높은 픽셀만을 선택하여 학습함으로써 타겟 도메인에 대한 적응력을 높였다.
3. **Model Soup를 통한 일반화 성능 향상**: 서로 다른 데이터 증강(Augmentation) 기법으로 학습된 여러 모델의 가중치를 산술 평균하는 Model Soup 기법을 적용하여 추론 비용의 증가 없이 성능을 최적화하였다.

## 📎 Related Works

본 연구는 $\text{DAFormer}$ 프레임워크를 베이스라인으로 사용한다. $\text{DAFormer}$는 트랜스포머 기반의 인코더와 다단계 문맥 인식 퓨전 디코더(Multi-level context-aware fusion decoder)를 통해 도메인 적응 시맨틱 세그멘테이션에서 우수한 성능을 보였다.

기존의 $\text{DAFormer}$는 $\text{SegFormer}$의 MiT 백본을 사용하지만, 본 논문은 사전 학습(Pre-training)의 규모와 네트워크 구조가 다운스트림 태스크의 성능에 결정적인 영향을 미친다는 선행 연구를 바탕으로 백본을 변경하였다. 또한, 일반적인 도시 교통 시나리오(Cityscapes 등)에서 효과적이었던 Rare Class Sampling 기법이 폐기물 분류 시나리오에서는 오히려 성능을 저하시킨다는 점을 발견하고 이를 제외함으로써 차별점을 두었다.

## 🛠️ Methodology

$\text{SIA\_Adapt}$의 전체 파이프라인은 초기 적응 모델 생성, 의사 레이블 생성, 자기 학습, 그리고 모델 수프(Model Soup) 단계로 구성된다.

### 1. 초기 적응 모델 ($\text{Initial Adaptive Model}, G_{init}$)

먼저 레이블이 있는 소스 데이터셋 $D_s$와 레이블이 없는 타겟 데이터셋 $D_t$를 사용하여 UDA 학습을 수행한다.

- **백본 구조**: ImageNet-22K로 사전 학습된 $\text{ConvNeXt-L}$을 사용한다. 이는 $\text{DAFormer}$의 MiT보다 더 강력한 사전 지식을 제공하며, 더 깊은 네트워크 구조가 도메인 적응 성능을 향상시킨다고 판단했기 때문이다.
- **디코더**: $\text{DAFormer}$의 디코더 설계를 그대로 유지한다.
- **학습 전략**: $\text{DAFormer}$의 전략을 따르되, 폐기물 데이터의 특성을 고려하여 $\text{Rare Class Sampling}$은 제외하였다.

### 2. Pseudo-labeling 및 Self-training

$G_{init}$을 사용하여 타겟 도메인 $D_t$에 대한 의사 레이블을 생성한다. 이때 발생할 수 있는 노이즈를 제거하기 위해 다음과 같은 신뢰도 임계값(Confidence threshold)을 적용한다.

- **필터링**: 예측 결과의 신뢰도가 $0.9$ 미만인 픽셀은 학습에서 제외한다.
- **다양한 증강 적용**: 정제된 의사 레이블을 바탕으로 세 가지 서로 다른 데이터 증강 기법($\text{PhotoMetricDistortion}$, $\text{GaussNoise}$, $\text{RandomGridShuffle}$)을 개별적으로 적용하여 세 개의 미세 조정 모델 $G_{aug\_a}, G_{aug\_b}, G_{aug\_c}$를 학습시킨다.

### 3. Model Soups

마지막으로, 추론 시 추가 연산 없이 성능을 높이기 위해 $\text{Model Soup}$ 기법을 적용한다. 이는 여러 모델의 가중치를 평균 내는 방식이다.

- **구성**: 초기 모델 $G_{init}$과 자기 학습을 통해 얻은 세 모델 $G_{aug\_a}, G_{aug\_b}, G_{aug\_c}$를 대상으로 한다.
- **결합**: $\text{Greedy soup}$ 방식을 통해 가중 평균을 수행하여 최종 모델 $G_{final}$을 생성한다.

$$G_{final} = \sum_{i \in \{init, a, b, c\}} w_i G_i$$
(여기서 $w_i$는 각 모델의 기여도를 결정하는 가중치이다.)

## 📊 Results

### 실험 설정

- **데이터셋**: 소스 도메인으로 $\text{ZeroWastev1}$, 타겟 도메인으로 $\text{ZeroWastev2}$를 사용하였다.
- **지표**: $\text{mIoU}$ (mean Intersection over Union) 및 $\text{Acc}$ (Accuracy)를 사용하였다.
- **환경**: NVIDIA RTX8000 GPU 1대를 사용하였으며, 초기 모델은 40,000 iteration, 미세 조정 모델은 10,000 iteration 동안 학습되었다.

### 정량적 결과

본 방법론은 VisDA 2022 챌린지에서 1위를 기록하였다.

- **최종 성능**: $\text{mIoU } 59.42$, $\text{Acc } 93.18$을 달성하여 2위 팀과의 상당한 격차를 보였다.
- **Source Only 성능**: 타겟 도메인 데이터에 접근하지 않고 소스 데이터만으로 학습했을 때도 $\text{mIoU } 56.46$을 기록하며 다른 팀들의 UDA 결과보다 높은 성능을 보였다. 이는 $\text{ConvNeXt-L}$과 $\text{ImageNet-22K}$ 사전 학습의 영향력이 매우 큼을 시사한다.
- **구성 요소별 기여도**: $\text{Model Soup}$ 적용 전보다 적용 후 $\text{mIoU}$가 $58.72 \rightarrow 59.42$로 상승하며 일반화 성능이 개선됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문의 가장 큰 성과는 복잡한 UDA 알고리즘의 설계보다 **강력한 백본과 대규모 사전 학습 데이터의 선택**이 실질적인 성능 향상에 더 결정적인 역할을 할 수 있음을 입증한 점이다. 특히 타겟 데이터 없이도 높은 성능을 낸 'Source Only' 결과는 도메인 전이(Transfer)의 기초가 되는 representation의 품질이 얼마나 중요한지를 보여준다.

### 한계 및 비판적 해석

- **데이터 특성 의존성**: $\text{Rare Class Sampling}$을 제거하여 성능을 높였다는 점은, 본 연구의 결과가 특정 데이터셋(폐기물)의 클래스 분포 특성에 강하게 의존하고 있음을 의미한다. 따라서 다른 도메인에 적용할 때는 해당 기법의 유지 여부를 다시 검토해야 한다.
- **모델 크기**: $\text{ConvNeXt-L}$은 성능은 좋으나 모델 파라미터 수가 많아 연산 자원이 제한적인 환경에서는 효율성이 떨어질 수 있다.

## 📌 TL;DR

본 논문은 **ImageNet-22K 사전 학습된 $\text{ConvNeXt-L}$ 백본**, **신뢰도 기반의 Pseudo-labeling**, 그리고 **Model Soup 가중치 평균화**를 결합한 $\text{SIA\_Adapt}$ 방법론을 제안하여 VisDA 2022 챌린지에서 1위를 차지하였다. 이 연구는 UDA 작업에서 네트워크 아키텍처와 대규모 사전 학습의 중요성을 강조하며, 향후 효율적인 도메인 적응 모델 설계의 방향성을 제시한다.
