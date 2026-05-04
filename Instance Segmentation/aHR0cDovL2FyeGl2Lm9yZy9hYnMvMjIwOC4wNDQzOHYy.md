# Occlusion-Aware Instance Segmentation via BiLayer Network Architectures

Lei Ke, Yu-Wing Tai, and Chi-Keung Tang (2023)

## 🧩 Problem to Solve

본 논문은 이미지 내에서 심하게 겹쳐 있는(highly-overlapping) 객체들을 정교하게 분할(segmentation)하는 문제를 해결하고자 한다. 일반적으로 인스턴스 분할 모델들은 객체의 실제 외곽선(real object contours)과 가려짐으로 인해 발생하는 경계선(occlusion boundaries)을 구분하는 데 어려움을 겪는다.

기존의 Mask R-CNN 및 그 변형 모델들은 각 인스턴스 마스크를 개별적으로 회귀(regress)하는 방식을 취한다. 이러한 접근 방식은 ROI(Region-of-Interest) 내의 객체가 거의 완전한 윤곽을 가지고 있다고 암묵적으로 가정하는데, 이는 COCO와 같은 일반적인 데이터셋에 심한 가려짐 사례가 적기 때문이다. 결과적으로 겹쳐 있는 객체들이 존재할 때, 모델은 가려진 영역을 제대로 처리하지 못하거나 경계선에서 오차가 발생하는 문제가 발생한다. 따라서 본 연구의 목표는 가려짐 관계를 명시적으로 모델링하여, 가리는 객체(occluder)와 가려지는 객체(occludee)를 분리하여 인식하는 Occlusion-aware Instance Segmentation 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 형성 과정을 두 개의 겹쳐진 레이어의 조합으로 모델링하는 **Bilayer Decoupling** 구조이다.

중심적인 직관은 ROI 내에서 가리는 객체(occluder)를 먼저 탐지하고, 이 정보를 바탕으로 가려진 객체(occludee)의 마스크를 추론하는 것이다. 이를 통해 단일 레이어에서 마스크를 직접 예측하던 기존 방식과 달리, 가리는 객체와 가려지는 객체의 경계를 자연스럽게 분리(decouple)하고, 마스크 회귀 과정에서 두 객체 간의 상호작용을 고려할 수 있게 된다. 이러한 구조는 GCN(Graph Convolutional Network) 기반의 BCNet뿐만 아니라 Vision Transformer(ViT) 기반의 구조로도 확장되어 적용되었다.

## 📎 Related Works

### 기존 연구 및 한계
1. **Two-stage Instance Segmentation**: Mask R-CNN, PANet, HTC 등은 바운딩 박스 탐지 후 마스크를 생성한다. 그러나 대부분의 개선이 백본 아키텍처에 집중되어 있으며, 가려짐이 심한 경우 마스크 회귀 단계에서 발생하는 충돌을 해결하기 위해 NMS나 후처리에 의존하는 경향이 있다.
2. **Amodal Instance Segmentation**: 가려진 부분까지 포함하여 전체 마스크를 예측하려는 시도(ASN, ORCNN 등)가 있었으나, 이들은 주로 단일 레이어 상에서 가려진 타겟 객체 하나만을 회귀시키므로 가리는 객체와 가려지는 객체 간의 상호작용 추론이 부족하다는 한계가 있다.
3. **Transformer-based Segmentation**: Mask2Former와 같은 쿼리 기반 모델들은 뛰어난 성능을 보이지만, 심한 가려짐 상황에서 개별 객체를 정교하게 구분하는 문제는 여전히 해결해야 할 과제로 남아 있다.

### 차별점
BCNet은 가리는 객체(occluder)와 가려지는 객체(occludee)를 서로 다른 그래프 레이어(또는 쿼리 그룹)로 명시적으로 분리한다. 단순히 타겟 객체의 마스크를 복원하는 것이 아니라, '가리는 객체의 형태와 위치'라는 힌트를 먼저 얻고 이를 통해 타겟 객체를 분할함으로써 더 강력한 가려짐 인식 및 추론 능력을 제공한다.

## 🛠️ Methodology

### 1. Bilayer GCN Structure (BCNet)
BCNet은 ROI 특징 추출 후 두 개의 GCN 레이어를 직렬로 배치한 구조를 가진다. GCN을 사용하는 이유는 픽셀 간의 비지역적(non-local) 관계를 고려하여, 가리는 객체에 의해 분리된 픽셀 영역들 사이에서도 정보를 전파하기 위함이다.

**그래프 컨볼루션 연산:**
그래프 $G=\langle V,E \rangle$에서 노드 $V$는 픽셀을 나타내며, 연산식은 다음과 같다.
$$Z = \sigma(AXW_g) + X$$
여기서 $X \in \mathbb{R}^{N \times K}$는 입력 특징, $A \in \mathbb{R}^{N \times N}$는 특성 유사도에 기반한 인접 행렬, $W_g$는 학습 가능한 가중치 행렬이다. 인접 행렬 $A$는 다음과 같이 dot product similarity로 계산된다.
$$A_{ij} = \text{softmax}(F(x_i, x_j)), \quad F(x_i, x_j) = \theta(x_i)^T \phi(x_j)$$

**Bilayer 흐름:**
1. **첫 번째 GCN (Occluder Layer):** 가리는 객체의 마스크와 외곽선(contour)을 동시에 예측한다. 이를 통해 가려짐 영역의 구체적인 형태와 위치 정보를 추출한다.
2. **특징 융합:** 첫 번째 GCN의 출력 $Z_0$를 입력 ROI 특징 $X_{roi}$에 더해 가려짐 인지 특징 $X_f$를 생성한다.
   $$X_f = Z_0 W_0^f + X_{roi}$$
3. **두 번째 GCN (Occludee Layer):** 융합된 특징 $X_f$를 입력받아 최종 타겟 객체인 가려지는 객체의 마스크와 외곽선을 예측한다.

### 2. Bilayer Transformer-based BCNet
Mask2Former를 기반으로 하며, 단일 디코더 구조를 **Bilayer Transformer Decoder**로 확장하였다.

- **Query 분리:** 학습 가능한 인스턴스 쿼리를 가리는 객체 쿼리(occluder queries)와 가려지는 객체 쿼리(occludee queries)의 두 그룹으로 나눈다. 가리는 객체 쿼리는 가려지는 객체 쿼리를 조건으로 하여 MLP를 통해 생성된다.
- **Cascaded Decoder:** 첫 번째 디코더가 가리는 객체의 정보를 먼저 추출하고, 이 정보가 잔차 연결(residual connection)을 통해 두 번째 디코더로 전달되어 가려지는 객체의 마스크 예측을 가이드한다.

### 3. 학습 목표 및 손실 함수
전체 네트워크는 다음과 같은 다중 작업 손실 함수를 통해 엔드-투-엔드로 학습된다.
$$L = \lambda_1 L_{Detect} + L_{Occluder} + L_{Occludee}$$
여기서 가리는 객체와 가려지는 객체 각각에 대해 외곽선 예측 손실($L_{Occ-B}$)과 마스크 분할 손실($L_{Occ-S}$)을 정의하며, 모두 Binary Cross-Entropy(BCE) 손실을 사용한다.
$$L'_{Occ-B} = L_{BCE}(W_B F_{occ}(X_{roi}), GT_B)$$
$$L'_{Occ-S} = L_{BCE}(W_S F_{occ}(X_{roi}), GT_S)$$

## 📊 Results

### 실험 설정
- **데이터셋:** COCO, KINS, COCOA, YTVIS, OVIS, BDD100K MOTS.
- **COCO-OCC:** 가려짐 성능을 정밀하게 평가하기 위해, 바운딩 박스 겹침 비율이 0.2 이상인 이미지들만 모은 서브셋을 자체적으로 구축하였다.
- **SOD (Synthetic Occlusion Dataset):** 가려짐 패턴을 다양화하기 위해 6만 개 이상의 단일 객체 뱅크(COB)를 활용하여 10만 장의 합성 가려짐 이미지를 생성하여 학습에 활용하였다.

### 주요 결과
1. **정량적 성능:**
   - **COCO:** Transformer 기반 BCNet은 ResNet-50-FPN 백본 사용 시 44.6 mask AP를 달성하여 Mask2Former(43.6 AP)를 상회하였다.
   - **COCO-OCC:** 가리는 객체 모델링을 추가했을 때 AP가 29.04에서 30.37로 상승하였으며, SOD 데이터셋으로 추가 학습 시 32.89 AP까지 향상되었다.
   - **Amodal Segmentation:** KINS 데이터셋에서 28.87 AP를 기록하며 기존 SOTA 모델들(ASN 등)보다 우수한 성능을 보였다.
2. **일반화 능력:**
   - **Object Detector:** FCOS, Faster R-CNN, Query-based detector 등 다양한 탐지기 모두에서 BCNet 구조를 적용했을 때 일관된 성능 향상이 나타났다.
   - **Video Instance Segmentation:** YTVIS에서 AP가 2.1 상승하였고, OVIS에서도 15.4에서 17.1로 성능이 크게 개선되었다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 가려짐 문제를 해결하기 위해 단순히 마스크를 '복원'하는 것이 아니라, '가리는 주체'를 먼저 인식하는 전략을 취했다. 실험 결과, GCN 기반의 bilayer 구조는 가려진 객체의 보이지 않는 부분을 더 합리적으로 추론하며, 특히 비슷한 색상과 형태를 가진 객체들이 겹쳐 있을 때 기존 모델보다 훨씬 명확하게 객체를 분리해내는 능력을 보여주었다. 또한 가리는 객체와 가려지는 객체의 예측 결과를 시각적으로 확인할 수 있어 모델의 판단 근거가 명확하다는 점에서 설명 가능성(explainability)이 높다.

### 한계 및 비판적 해석
1. **미학습 클래스 문제:** 학습 데이터에 없던 새로운 클래스의 객체가 가리는 객체로 등장할 경우, 첫 번째 레이어의 예측이 부정확해지며 결과적으로 전체 성능이 기존의 단일 레이어 모델 수준으로 저하될 가능성이 있다.
2. **탐지기 의존성:** 본 모델은 ROI 특징을 기반으로 하므로, 1단계 바운딩 박스 탐지기의 정확도가 낮을 경우 마스크 예측 성능이 함께 제한될 수밖에 없다.
3. **비디오 데이터 활용 미흡:** 비디오 인스턴스 분할에서도 성능 향상을 보였으나, 이는 프레임별 마스크 헤드를 교체한 결과일 뿐 비디오의 핵심인 시간적 정보(temporal cues)를 직접적으로 활용하여 가려짐을 해결하는 메커니즘은 포함되지 않았다.

## 📌 TL;DR

본 논문은 심한 가려짐이 있는 객체 분할 문제를 해결하기 위해 **가리는 객체(Occluder)**와 **가려지는 객체(Occludee)**를 분리하여 처리하는 **Bilayer Decoupling** 구조의 BCNet을 제안한다. 이 구조는 GCN과 Transformer 디코더 모두에 적용 가능하며, 가리는 객체의 정보를 가이드로 삼아 가려진 객체의 마스크를 정교하게 예측함으로써 COCO, KINS 등 주요 벤치마크에서 SOTA 성능을 달성하였다. 특히 합성 가려짐 데이터셋(SOD)의 구축과 활용을 통해 모델의 강건성을 높였으며, 향후 가려짐 인식 기반의 인스턴스 분할 연구에 중요한 방법론적 기틀을 제공할 것으로 기대된다.