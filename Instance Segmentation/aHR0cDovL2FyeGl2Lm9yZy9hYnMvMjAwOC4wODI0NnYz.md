# Salient Instance Segmentation with Region and Box-level Annotations

Jialun Pei, He Tang, Tianyang Cheng and Chuanbo Chen (Year not specified)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Salient Instance Segmentation (SIS) 작업을 수행하기 위한 고품질의 픽셀 수준(pixel-level) 학습 데이터 부족 문제이다.

전통적인 Salient Object Detection (SOD)은 이미지 내에서 가장 눈에 띄는 영역을 이진 마스크(binary mask) 형태로 찾는 영역 수준(region-level)의 작업이었다. 반면, SIS는 각 개별 인스턴스를 식별하고 정밀한 마스크를 생성해야 하는 인스턴스 수준(instance-level)의 작업으로, 훨씬 더 상세한 분석이 가능하다. 그러나 SIS를 위한 픽셀 단위의 정밀한 주석(annotation)을 생성하는 작업은 비용이 매우 많이 들고 시간이 오래 걸리며, 현재 가용한 데이터셋의 규모 또한 제한적이다.

따라서 본 논문의 목표는 정밀한 픽셀 수준의 레이블링 없이, 상대적으로 획득하기 쉬운 Bounding Box와 SOD의 이진 영역(Binary region) 정보만을 결합한 '부정확한 감독(inexact supervision)'을 통해 성능이 뛰어난 SIS 프레임워크를 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **CGCNet (Cyclic Global Context Network) 제안**: 정밀한 마스크 대신, 기존 SOD 데이터셋에서 얻을 수 있는 Salient region과 Bounding box를 결합하여 학습하는 새로운 SIS 프레임워크를 제안하였다.
2. **GFR (Global Feature Refining) 레이어 설계**: Mask R-CNN의 ROIAlign이 수용 영역(receptive field)을 국소적으로 제한하는 것과 달리, GFR은 각 인스턴스의 특징을 전역 문맥(global context)으로 확장한다. 이를 통해 배경 특징을 충분히 활용하고 다른 인스턴스로부터의 간섭을 억제하여 경계 정보를 더 세밀하게 포착한다.
3. **레이블 업데이트 스킴 (Labeling Updating Scheme) 도입**: 학습 과정에서 예측된 마스크와 Conditional Random Field (CRF)를 이용하여 초기 단계의 거친 레이블(coarse-grained labels)을 반복적으로 최적화하는 순환 구조를 설계하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 분석하고 차별점을 제시한다.

1. **Salient Object Detection (SOD)**: 딥러닝의 발전으로 SOD는 비약적인 발전을 이루었으나, 대부분 결과물이 영역 수준의 이진 마스크에 그쳐 개별 인스턴스를 구분하지 못한다는 한계가 있다.
2. **Salient Instance Segmentation (SIS)**: 기존의 SIS 연구들은 주로 Fully-supervised 방식으로 진행되었으며, 최근 단일 단계(single-shot) 프레임워크 등이 제안되었다. 그러나 데이터셋의 부족으로 인해 모델의 성능 향상이 제한되는 병목 현상이 발생하고 있다.
3. **Weakly Supervised Learning**: 레이블링 비용을 줄이기 위해 이미지 수준의 레이블이나 불완전한 레이블을 사용하는 연구들이 진행되어 왔다. 본 논문은 이러한 흐름에 착안하여 SOD의 영역 정보와 Bounding Box라는 두 가지 약한 감독 소스를 결합하여 SIS에 적용함으로써 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

CGCNet은 end-to-end 방식의 2단계 프레임워크이다. 먼저 RPN(Region Proposal Network) head를 통해 Salient proposal(Bounding Box)을 탐지하고, 이후 GFR 모듈과 SIS 브랜치를 통해 각 proposal에 대한 정밀한 픽셀 수준의 마스크를 예측한다. 백본 네트워크로는 ResNet-101과 FPN(Feature Pyramid Network)을 사용한다.

### 부정확한 감독 소스 (Inexact Supervision Sources)

정밀한 마스크가 없는 상황에서 학습을 위해, SOD 데이터셋(DUT-OMRON)의 이진 맵 $S$와 Bounding box $W_i$를 결합하여 거친 레이블(coarse-grained label) $I$를 생성한다. 이때, 한 박스 내에 여러 영역이 포함되는 문제를 해결하기 위해 '하나의 박스에는 하나의 최대 영역만 유지한다'는 사전 지식(prior)을 적용한다. 최종 레이블 $I$는 다음과 같이 정의된다.

$$I = \sum_{i=1}^{n} [S(x,y) \cap W_i - \phi_i(\hat{x}, \hat{y})]$$

여기서 $S(x,y)$는 이진 살리언시 맵, $W_i$는 각 인스턴스의 박스, $\phi_i$는 사전 지식에 의해 제외된 픽셀 집합이며, $n$은 이미지 내 인스턴스의 수이다.

### GFR (Global Feature Refining) 모듈

GFR의 목적은 국소적인 ROI 특징의 한계를 넘어 전역 문맥 정보를 획득하고, 다른 인스턴스의 특징 간섭을 줄이는 것이다. FPN에서 추출된 입력 특징 맵을 $F$, $n$개의 proposal 각각의 특징 맵을 $R_i$라고 할 때, $i$번째 인스턴스를 위한 전역 정제 특징 $G_i$는 다음과 같이 계산된다.

$$G_i = F - \sum_{j=1}^{n} R_j + R_i, \quad i=1, 2, \dots, n$$

이 수식은 전체 특징 맵 $F$에서 모든 proposal 영역의 특징을 제거한 뒤, 현재 분석 대상인 $i$번째 특징만을 다시 더함으로써, '현재 인스턴스 + 배경'의 구조를 만들어 Center-surround contrast를 극대화한다.

### 레이블 업데이트 스킴 및 CRF

초기 거친 레이블의 경계가 부정확한 문제를 해결하기 위해 CRF를 적용하여 예측 마스크 $R$을 정제된 마스크 $R^f$로 변환한다. 이후 KL-Divergence를 이용하여 기존 레이블 $C$를 업데이트할지 결정한다.

$$K(R, C) = \frac{1}{H \times W} \sum_{i=1}^{H \times W} C_i \log \left( \frac{C_i}{R_i + \sigma} + \sigma \right)$$

알고리즘 흐름은 다음과 같다.

1. 예측 값 $R$과 CRF 정제 값 $R^f$에 대해 각각 $K_1(R, C)$와 $K_2(R^f, C)$를 계산한다.
2. 만약 $K_2(R^f, C) - K_1(R, C) \ge \phi$ (임계값 0.05)라면, 기존 레이블 $C$를 유지한다.
3. 그렇지 않다면, 정제된 예측 값 $R^f$로 레이블을 업데이트하여 다음 반복 학습에 사용한다.

### 손실 함수 (Loss Function)

전체 손실 함수는 Bounding Box 손실, 세그멘테이션 손실, 그리고 업데이트 관련 손실의 합으로 구성된다.

$$L = L_{bb} + L_{seg} + L_{upd}$$

여기서 $L_{seg}$는 픽셀별 클래스 확률에 대한 Cross-entropy loss를 사용하며, $L_{upd}$는 위에서 언급한 KL-Divergence의 차이값으로 정의된다.

## 📊 Results

### 실험 설정

- **데이터셋**: DUT-OMRON (학습), Dataset1K (테스트), SOC (속성별 테스트).
- **지표**: $AP^r$ (IoU 0.5 및 0.7 기준 정밀도 평균) 및 $AP$ (IoU 0.5~0.95까지 0.05 간격으로 평균한 값).
- **비교 대상**: MSRNet, SCNet, S4Net 등 최신 SIS 방법론.

### 주요 결과

1. **정량적 성과**: Dataset1K 테스트 셋에서 CGCNet은 부정확한 레이블로 학습했음에도 불구하고 Mask AP 58.3%를 달성하여, Fully-supervised 방식인 S4Net 등을 능가하는 성능을 보였다.
2. **어블레이션 연구 (Ablation Study)**:
    - **Backbone**: ResNeXt-101이 가장 높은 성능을 보였으며, ResNet-101이 그 뒤를 이었다.
    - **GFR 모듈**: ROIAlign이나 ROIMasking보다 GFR이 $AP^r_{0.7}$ 지표에서 월등히 높은 성능을 보여, 전역 문맥 활용의 중요성을 입증하였다.
    - **구성 요소 기여도**: 업데이트 스킴(US)이 AP 성능을 약 2% 향상시켰으며, 사전 지식(PC) 기반 레이블링이 $AP^r$ 지표 향상에 크게 기여하였다.
3. **SOC 데이터셋 분석**: 복잡한 배경(clutter) 환경에서도 GFR 모듈 덕분에 배경 억제 능력이 향상되어 다양한 속성(Appearance Change, Heterogeneous Object 등)에서 강건한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 SIS 작업에서 가장 큰 병목이었던 '고비용의 픽셀 수준 레이블' 문제를 효과적으로 우회하였다. 단순히 약한 감독을 사용하는 것에 그치지 않고, GFR을 통한 전역 특징 추출과 CRF 기반의 순환적 레이블 업데이트라는 구조적 장치를 통해 Fully-supervised 모델에 근접하거나 이를 능가하는 성능을 낸 점이 매우 고무적이다.

### 한계 및 비판적 해석

논문에서 제시한 Failure Mode 분석에 따르면 다음과 같은 한계가 존재한다.

1. **미세 특징 포착 부족**: 매우 가느다란 국소적 특징에 대해서는 민감도가 떨어진다.
2. **인스턴스 개수 예측 오류**: 2단계 구조와 NMS 의존성으로 인해 실제보다 더 많은 인스턴스를 예측하는 경우가 발생한다.
3. **중첩 영역의 경계 모호성**: 두 인스턴스가 겹쳐 있을 때 경계가 뭉개지는 현상이 나타나는데, 이는 초기 학습 데이터가 Bounding Box 기반의 거친 레이블이었기 때문에 발생하는 근본적인 한계로 보인다.

## 📌 TL;DR

본 논문은 고비용의 픽셀 마스크 대신 Bounding Box와 SOD 이진 맵만을 활용해 학습하는 **CGCNet**을 제안하였다. **GFR 모듈**을 통해 전역 문맥을 활용함으로써 인스턴스 구분을 명확히 하고, **CRF 기반의 순환 업데이트**를 통해 불완전한 레이블을 점진적으로 정교화하였다. 그 결과, 매우 적은 레이블링 비용으로도 기존의 Fully-supervised SIS 모델들을 뛰어넘는 성능을 달성하였으며, 이는 향후 대규모 데이터셋 구축이 어려운 비디오 감시 시스템 등의 실제 적용 분야에서 매우 중요한 역할을 할 것으로 기대된다.
