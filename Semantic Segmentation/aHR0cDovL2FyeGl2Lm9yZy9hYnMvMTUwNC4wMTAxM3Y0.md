# Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation

Guosheng Lin, Chunhua Shen, Anton van den Hengel, Ian Reid

## 🧩 Problem to Solve

본 논문은 시맨틱 이미지 분할(Semantic Image Segmentation)을 개선하는 것을 목표로 합니다. 최근 딥 컨볼루션 신경망(CNN)을 이용한 분할 연구에서 뛰어난 성과를 보였지만, 이미지 영역 간의 '패치-패치(patch-patch)' 문맥과 '패치-배경(patch-background)' 문맥과 같은 중요한 문맥 정보를 명시적으로 모델링하는 데 부족함이 있었습니다. 특히, CNN 기반의 Pairwise Potential 함수를 갖는 Conditional Random Fields (CRFs)를 사용하여 패치 간의 시맨틱 상관관계를 포착하는 것은 계산적으로 매우 비쌉니다. 학습 과정에서 역전파(backpropagation)를 위해 반복적으로 고비용의 CRF 추론을 수행해야 하기 때문입니다.

## ✨ Key Contributions

- **CNN 기반 Pairwise Potential 함수를 CRF에 통합:** 이미지 패치 간의 시맨틱 관계를 명시적으로 모델링하기 위해 CNN 기반의 일반적인 Pairwise Potential 함수를 CRF에 통합하는 방법을 제안합니다.
- **효율적인 구분 학습(Piecewise Training) 기법 적용:** 심층 CNN-CRF 공동 학습의 효율성 문제를 해결하기 위해 CRF의 구분 학습 기법을 적용합니다. 이를 통해 확률적 경사 하강법(SGD) 반복마다 발생하는 반복적인 고비용 추론을 피하여 효율적인 학습을 가능하게 합니다.
- **다중 스케일 입력 및 슬라이딩 피라미드 풀링을 통한 배경 문맥 활용:** 전통적인 다중 스케일 이미지 입력 방식과 슬라이딩 피라미드 풀링(sliding pyramid pooling)을 활용하는 네트워크 아키텍처를 통해 패치-배경 문맥 정보를 효과적으로 인코딩합니다. 이 방법의 성능 향상 효과를 경험적으로 입증합니다.
- **최첨단 성능 달성:** NYUDv2, PASCAL VOC 2012, PASCAL-Context, SIFT-flow 등 여러 유명 시맨틱 분할 데이터셋에서 새로운 최첨단 성능을 달성했습니다. 특히, PASCAL VOC 2012 데이터셋에서 78.0의 Intersection-over-Union (IoU) 점수를 기록했습니다.

## 📎 Related Works

- **문맥 정보 활용:** [39, 21, 7]과 같은 초기 연구들은 이미지 이해를 위한 공간 문맥의 중요성을 강조했습니다. 특히 "TAS" [21]는 생성적 확률론적 그래픽 모델을 사용하여 Things(객체)와 Stuff(배경) 간의 다양한 공간 문맥을 모델링했습니다.
- **CNN 기반 분할 방법:** 최근 가장 성공적인 방법들은 CNN에 기반하며, [14, 19]와 같은 영역 제안(region proposal) 기반 방법과 [32, 3, 5]의 Fully Convolutional Neural Networks (FCNNs)가 있습니다. FCNN은 효율성과 종단 간(end-to-end) 학습 능력으로 인해 인기를 얻었습니다.
- **저해상도 예측 정제:** FCNN 기반 방법의 저해상도 예측 문제를 해결하기 위해 많은 연구가 진행되었습니다. DeepLab-CRF [3]는 밀집 CRF(dense CRF) [24]를 사용하여 경계를 정제하고, CRF-RNN [48]은 이를 순환 레이어(recurrent layers)로 확장하여 종단 간 학습을 가능하게 했습니다. 이 외에도 Deconvolution layers [35], Super-pixel pooling [30], coarse-to-fine 학습 [9], 중간 레이어 특징(skip connections) 활용 [18, 32] 등이 있습니다. 본 논문의 방법은 이러한 정제 단계와 상보적(complementary)으로 작동합니다.
- **CNN과 CRF 결합:** [3, 48, 40] 등은 CNN과 CRF의 강점을 결합하는 데 초점을 맞추었으며, 대부분은 경계 스무딩(smoothness)을 위한 Potts-모델 기반 Pairwise Potential을 사용했습니다. 본 논문은 이러한 방법들과 달리, 이미지 영역 간의 시맨틱 호환성을 모델링하기 위해 CNN 기반의 Pairwise Potential 함수를 학습합니다.
- **효율적인 CRF 학습:** CRF 학습의 고비용 문제를 해결하기 위해 유사우도(pseudo-likelihood) 학습 [1] 및 구분 학습(piecewise learning) [43]과 같은 근사 학습 방법이 제안되었습니다. 본 연구는 [43, 23]에서 사용된 구분 학습 개념을 CNN 포텐셜(potential) 학습에 적용합니다.

## 🛠️ Methodology

본 논문은 심층 구조 모델(Deep Structured Model)을 제안하며, 이는 Multi-scale CNN 기반의 Unary 및 Pairwise Potential을 Conditional Random Field (CRF)에 결합하고, 효율적인 학습을 위해 구분 학습(Piecewise Training)을 사용합니다.

1. **FeatMap-Net:**

   - 입력 이미지를 먼저 'FeatMap-Net'이라는 컨볼루션 네트워크를 통과시켜 특징 맵(feature map)을 생성합니다. 이 특징 맵은 원본 이미지보다 낮은 해상도를 가집니다.
   - **배경 문맥 활용:**
     - **다중 스케일 입력:** 입력 이미지를 3가지 스케일(예: 1.2, 0.8, 0.4)로 크기를 조절하여 네트워크에 입력합니다. 상위 5개의 컨볼루션 블록은 모든 스케일에서 공유되며, 각 스케일에는 전용 컨볼루션 블록(Conv Block 6)이 있습니다.
     - **슬라이딩 피라미드 풀링:** 생성된 특징 맵에 2단계 슬라이딩 피라미드 풀링(sliding pyramid pooling, 5x5 및 9x9 슬라이딩 맥스 풀링 윈도우)을 적용하여 다양한 크기의 배경 영역에서 정보를 캡처하고 특징 맵의 시야(field-of-view)를 넓힙니다.
   - 초기 가중치는 VGG-16 네트워크에서 전이 학습(transfer learning)을 통해 얻습니다.

2. **CRF 그래프 구성:**

   - FeatMap-Net의 특징 맵 각 위치(원하는 입력 이미지의 사각형 영역에 해당)를 CRF 그래프의 노드(node)로 만듭니다.
   - **Pairwise 연결:** CRF 그래프의 노드들은 공간 범위 상자(spatial range box) 내에 있는 다른 모든 노드와 연결됩니다 (예: "주변" 및 "상하" 관계).

3. **Potential 함수 정의:**

   - **Unary Potential 함수:** $U(y_p,x_p; \theta_U) = -z_{p,y_p}(x; \theta_U)$
     - FeatMap-Net이 생성한 특징 맵과 얕은 Fully Connected Network (Unary-Net)를 통해 각 노드의 K 클래스에 대한 K차원 출력을 생성합니다.
   - **Pairwise Potential 함수:** $V(y_p,y_q,x_{pq}; \theta_V) = -z_{p,q,y_p,y_q}(x; \theta_V)$
     - 연결된 두 노드의 특징 벡터를 연결(concatenate)하여 엣지(edge) 특징을 형성합니다.
     - 이 엣지 특징을 얕은 Fully Connected Network (Pairwise-Net)에 입력하여 Pairwise Potential을 생성합니다. 이는 $K \times K$ 크기로, 가능한 모든 라벨 조합에 대한 신뢰도 값을 출력하며 노드 쌍의 시맨틱 호환성을 측정합니다.
     - Potts 모델과 달리, 이는 이웃 스무딩보다는 시맨틱 호환성을 모델링합니다.

4. **CRF 학습 (구분 학습):**

   - 전역 파티션 함수(global partition function) $Z(x;\theta)$의 계산과 그 기울기(gradient) 계산의 복잡성으로 인해 직접적인 최대 우도(maximum likelihood) 학습은 비효율적입니다.
   - 이를 해결하기 위해 **구분 학습(Piecewise Training)**을 적용합니다.
   - 조건부 우도(conditional likelihood)를 Unary 및 Pairwise Potential에 대해 정의된 독립적인 우도들의 곱으로 재구성합니다:
     $$P(y|x) = \prod_{U \in \mathcal{U}} \prod_{p \in N_U} P_U(y_p|x) \prod_{V \in \mathcal{V}} \prod_{(p,q) \in S_V} P_V(y_p,y_q|x)$$
   - $P_U(y_p|x)$는 K개 클래스에 대한 소프트맥스(Softmax) 정규화 함수이고, $P_V(y_p,y_q|x)$는 $K \times K$ 가능한 라벨 조합에 대한 소프트맥스 함수입니다.
   - 이를 통해 비용이 많이 드는 추론 없이 쉽게 기울기를 계산할 수 있으며, 포텐셜 함수들의 병렬 학습이 가능해집니다.

5. **예측 단계:**
   - **저해상도 예측 (Coarse-level Prediction):** 제안된 심층 구조 모델에서 CRF 추론(평균장 근사법(mean field approximation)으로 3회 반복)을 수행하여 저해상도(입력 이미지 크기의 1/16) 예측을 얻습니다.
   - **예측 정제 단계 (Prediction Refinement):**
     - 저해상도 예측의 점수 맵(score map)을 입력 이미지 크기로 이중선형 업샘플링(bilinear upsampling)합니다.
     - 객체 경계를 선명하게 하기 위해 밀집 CRF [24]와 같은 일반적인 후처리(post-processing) 방법을 적용하여 최종 고해상도 예측을 생성합니다. 이 과정은 픽셀 강도(색상 대비) 정보를 활용합니다.

## 📊 Results

제안된 방법은 4개의 주요 시맨틱 분할 데이터셋에서 평가되었으며, 모든 데이터셋에서 최첨단 성능을 달성했습니다.

- **NYUDv2 (40개 클래스, RGB 이미지 사용):**

  - 기존 방법인 FCN-32s (29.2 IoU) 및 다른 RGB-D 기반 방법들을 크게 능가하는 40.6 IoU를 달성하여 새로운 최첨단 기록을 세웠습니다.
  - **구성 요소별 평가 (Ablation Study):**
    - 기본 FCN 구현(FullyConvNet Baseline): 30.5 IoU
    - - 슬라이딩 피라미드 풀링: 32.4 IoU
    - - 다중 스케일: 37.0 IoU
    - - 경계 정제: 38.3 IoU
    - - CNN 문맥적 Pairwise Potential: 40.6 IoU (최종 성능)
    - 각 구성 요소가 성능 향상에 기여함을 입증했습니다.

- **PASCAL VOC 2012 (20개 객체 + 1개 배경 클래스):**

  - VOC 훈련 데이터만 사용: 75.3 IoU (VOC 훈련 데이터만 사용하는 방법 중 최고)
  - VOC + COCO 추가 훈련 데이터 사용: 77.2 IoU
  - **+ 중간 레이어 특징을 활용한 정제 레이어 및 경계 정제:** 78.0 IoU (도전적인 이 데이터셋에서 최고 기록된 결과)
  - 대부분의 개별 카테고리에서 경쟁 방법들보다 우수한 성능을 보였습니다.

- **PASCAL-Context (60개 클래스):**

  - 43.3 IoU를 달성하여 이 데이터셋에서 최고 성능을 기록했습니다.

- **SIFT-flow (33개 클래스):**
  - 44.9 IoU를 달성하여 이 데이터셋에서 최고 성능을 기록했습니다.

## 🧠 Insights & Discussion

- **문맥 정보의 중요성:** 본 연구는 이미지 패치 간의 시맨틱 관계와 패치-배경 문맥을 명시적으로 모델링하는 것이 시맨틱 분할 성능을 크게 향상시킬 수 있음을 입증했습니다. CNN 기반 Pairwise Potential은 기존의 경계 스무딩 목적의 포텐셜과 달리 시맨틱 호환성을 효과적으로 학습합니다.
- **구분 학습의 실용성:** 심층 구조 모델의 학습에서 발생하는 계산적 비효율성(CRF 추론 비용)을 구분 학습이라는 근사 학습 방법을 통해 성공적으로 해결했습니다. 이는 대규모 데이터에 대한 CNN-CRF 모델 학습을 실용적으로 만들었습니다.
- **다중 스케일 및 피라미드 풀링의 효과:** FeatMap-Net에 적용된 다중 스케일 이미지 입력과 슬라이딩 피라미드 풀링이 배경 정보를 효과적으로 인코딩하여 전반적인 분할 성능 향상에 중요한 역할을 한다는 것을 보여주었습니다.
- **한계 및 향후 연구:** 현재 시스템의 예측 정제 단계는 비교적 간단한 이중선형 업샘플링과 밀집 CRF 후처리를 사용합니다. 저자들은 이 단계가 Deconvolution Networks [35], 다중 Coarse-to-Fine 학습 네트워크 [9], 또는 중간 레이어 특징 활용 [18, 32]과 같은 보다 정교한 정제 방법들을 적용함으로써 추가적인 성능 향상을 얻을 수 있다고 언급합니다.
- **광범위한 적용 가능성:** 제안된 방법론은 시맨틱 분할 외에도 다른 컴퓨터 비전 태스크에 잠재적으로 광범위하게 적용될 수 있습니다.

## 📌 TL;DR

시맨틱 분할에서 문맥 정보의 중요성을 강조하며, 이를 위해 CNN 기반 Unary 및 Pairwise Potential을 활용하는 심층 구조 CRF 모델을 제안합니다. 특히, 고비용의 CRF 추론을 피하기 위해 **구분 학습(Piecewise Training)** 방식을 도입하여 효율적인 학습을 가능하게 했습니다. 또한 다중 스케일 입력과 슬라이딩 피라미드 풀링으로 패치-배경 문맥을 효과적으로 포착합니다. 결과적으로 NYUDv2, PASCAL VOC 2012 등 여러 벤치마크 데이터셋에서 **최첨단 성능을 달성**하며, 명시적인 문맥 모델링의 효과를 입증했습니다.
