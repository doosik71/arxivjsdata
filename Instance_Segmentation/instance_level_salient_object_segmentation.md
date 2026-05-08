# Instance-Level Salient Object Segmentation

Guanbin Li, Yuan Xie, Liang Lin, Yizhou Yu

## 🧩 Problem to Solve

기존의 객체 salient (가장 두드러지는) 영역 감지(salient region detection) 연구들은 이미지 내에서 시각적으로 두드러지는 픽셀들을 찾는 데 집중했지만, 이 픽셀들이 **개별적인 객체 인스턴스(individual object instances)**에 속하는지 구분하지 못했습니다. 이는 "salient region detection"이라고 불리며, 여러 실용적인 응용 분야(예: 이미지 캡셔닝, 멀티라벨 이미지 인식)에서 더 상세한 객체 인스턴스 정보를 요구하는 한계를 가집니다. 본 논문은 이러한 한계를 해결하고, 감지된 salient 영역 내에서 **개별적인 객체 인스턴스를 식별**하는 "인스턴스 수준 salient 객체 분할(instance-level salient object segmentation)"이라는 새로운 도전 과제를 제시합니다.

## ✨ Key Contributions

* **다중 스케일 정제 네트워크 (MSRNet) 개발**: salient 영역 감지를 위한 완전 합성곱(Fully Convolutional) 다중 스케일 정제 네트워크인 MSRNet을 개발했습니다. MSRNet은 상향식(bottom-up) 및 하향식(top-down) 정보를 통합하며, 동일 이미지의 다른 스케일 버전을 고려하여 픽셀 수준의 가중치를 자동 학습함으로써 기존 방법보다 훨씬 높은 정밀도를 달성합니다.
* **Salient 객체 윤곽선 감지 및 인스턴스 분할**: MSRNet은 salient 객체 윤곽선 감지에도 잘 일반화되어, 감지된 salient 영역 내에서 개별 객체 인스턴스를 분리할 수 있게 합니다. 객체 제안(object proposal) 생성 및 필터링 기술과 통합하여 고품질의 분할된 salient 객체 인스턴스를 생성합니다.
* **새로운 데이터셋 구축**: salient 인스턴스 분할 연구 및 평가를 위한 1,000개의 이미지와 픽셀 단위 salient 인스턴스 주석을 포함하는 새로운 벤치마크 데이터셋을 구축했습니다. MSRNet을 사용하여 salient 윤곽선 감지 및 salient 인스턴스 분할에 대한 벤치마크 결과를 제공합니다.

## 📎 Related Works

* **Salient 영역 감지 (Salient Region Detection)**: 초기 저수준 특징 기반(bottom-up) 및 고수준 지식 통합(top-down) 방법에서 발전하여, 최근에는 심층 CNN 기반 방법들이 정확도를 크게 향상시켰습니다. 그러나 대부분의 심층 CNN 기반 방법들은 단일 스케일 입력 이미지를 다루므로, 다중 스케일 객체에 대한 성능이 저하될 수 있습니다.
* **객체 제안 (Object Proposals)**: 객체 제안은 최소한의 가설로 타겟 객체를 지역화하는 것을 목표로 합니다. 객체성(objectness) 점수에 따라 순위를 매기는 방법이나, 다단계 분할 결과를 병합하여 제안을 생성하는 방법이 있습니다. 하지만 이러한 방법들은 일반적으로 salient 객체 지역화에 특화되어 있지 않습니다.
* **인스턴스 인식 의미론적 분할 (Instance-Aware Semantic Segmentation)**: 객체 감지와 의미론적 분할을 통합하는 작업으로, 다중 작업 학습(multi-task learning) 문제 또는 종단 간(end-to-end) 통합 모델로 접근되었습니다. 본 연구는 이러한 작업에서 영감을 받아, 미리 정의된 의미 범주에 얽매이지 않고 salient 객체와 그 인스턴스를 동시에 탐지하는 "salient 인스턴스 분할"을 제안합니다.

## 🛠️ Methodology

본 논문의 salient 인스턴스 분할 방법은 크게 네 가지 단계로 구성됩니다:

1. **Salient 영역 및 객체 윤곽선 감지 (MSRNet)**:
    * **정제된 VGG 네트워크**: VGG16 네트워크를 기반으로 상향식 특징(하위 계층)과 하향식 정제(상위 계층)를 통합합니다. 두 개의 완전 연결 계층을 1x1 합성곱 계층으로 변환하고, 마지막 두 풀링 계층에서 서브샘플링을 건너뛰어 특징 맵 해상도를 높이며, atrous convolution을 사용하여 원래의 수용장(receptive field)을 유지합니다. 풀링 계층에 대응하는 5개의 정제 모듈을 통해 저수준 특징과 고수준 의미 정보를 결합하여 입력 이미지와 동일한 해상도의 출력 맵을 생성합니다.
    * **다중 스케일 주의 가중치 융합 (Multiscale Fusion with Attentional Weights)**: 3개의 정제된 VGG 네트워크 스트림을 복제하여 입력 이미지를 세 가지 다른 스케일($s \in \{1, 0.75, 0.5\}$)로 처리합니다. 각 스트림의 출력 확률 맵($M_{s}^{c}$)은 입력 이미지와 동일한 해상도로 조정됩니다. 학습된 **주의 모듈(attention module)**은 각 픽셀 위치와 스케일에 대한 soft 가중치($W_s$)를 생성하여 이 맵들을 픽셀 단위로 가중합하여 최종 맵($F_c = \Sigma_{s \in \{1,0.75,0.5\}} W_s \cdot M_{s}^{c}$)을 계산합니다.
    * **학습**: salient 영역 감지 및 salient 객체 윤곽선 감지를 위해 동일한 MSRNet 아키텍처를 기반으로 두 개의 심층 모델을 학습시킵니다. 불균형한 픽셀 수 문제를 해결하기 위해 손실 함수에 가중치를 부여합니다.

2. **Salient 인스턴스 제안 (Salient Instance Proposal)**:
    * MCG (Multiscale Combinatorial Grouping) 알고리즘을 사용하여 MSRNet이 감지한 salient 객체 윤곽선으로부터 객체 제안을 생성합니다.
    * 초기 800개의 객체 제안 중 salient 픽셀 비율이 80% 미만인 제안은 제거합니다.
    * 이후 MAP-기반 부분집합 최적화(MAP-based subset optimization)를 적용하여 중복되거나 노이즈가 많은 제안을 걸러내고, 최종적으로 압축된 salient 객체 인스턴스 세트를 얻습니다.

3. **Salient 인스턴스 분할 정제 (Refinement of Salient Instance Segmentation)**:
    * 감지된 salient 인스턴스 수($K$)와 배경을 포함하여 $K+1$개의 클래스를 가지는 멀티클래스 레이블링 문제로 간주합니다.
    * 초기 salient 인스턴스 확률 맵을 기반으로, 완전 연결 CRF (Conditional Random Field) 모델을 사용하여 공간적 일관성 및 윤곽선 지역화를 개선합니다.
    * CRF 에너지 함수는 단항 포텐셜(unary potential)과 쌍 포텐셜(pairwise potential)로 구성되며, 쌍 포텐셜은 픽셀 위치와 강도에 기반한 두 개의 가우시안 커널을 사용하여 유사한 색상의 인접 픽셀이 유사한 레이블을 가지도록 장려합니다.
    $$E(x) = -\sum_{i} \log P(x_i) + \sum_{i,j} \theta_{ij}(x_i,x_j)$$
    여기서 $\theta_{ij}(x_i,x_j)$는 다음과 같이 정의됩니다.
    $$\theta_{ij} = \mu(x_i,x_j) \left[ \omega_1 \exp \left( -\frac{\|p_i-p_j\|^2}{2\sigma_\alpha^2} - \frac{\|I_i-I_j\|^2}{2\sigma_\beta^2} \right) + \omega_2 \exp \left( -\frac{\|p_i-p_j\|^2}{2\sigma_\gamma^2} \right) \right]$$

## 📊 Results

* **Salient 영역 감지**: MSRNet은 MSRA-B, PASCAL-S, DUT-OMRON, HKU-IS, ECSSD, SOD 등 6개의 벤치마크 데이터셋에서 기존 8개 최신 방법(GC, DRFI, LEGS, MC, MDF, DCL+, DHSNet, RFCN)을 일관되게 능가하는 최첨단 성능을 달성했습니다.
  * 최대 F-measure에서 기존 최고 알고리즘 대비 1.33%에서 3.70%까지 향상되었습니다.
  * MAE (Mean Absolute Error)에서도 기존 최고 MAE를 8.5%에서 20.4%까지 감소시켰습니다.
  * MSRNet의 각 구성 요소(정제 모듈, 주의 기반 다중 스케일 융합)의 효과성이 개별적으로 검증되었습니다.
* **Salient 객체 윤곽선 감지 및 인스턴스 분할**: 새로 구축된 데이터셋에서 MSRNet 기반 프레임워크는 Salient 윤곽선 감지에서 ODS 0.719, OIS 0.757, AP 0.765의 높은 성능을 보였습니다.
  * Salient 인스턴스 분할에서는 $mAP_{\text{r}}@0.5$ (IoU 임계값 0.5)에서 65.32%, $mAP_{\text{r}}@0.7$에서 52.18%를 달성하여 강력한 벤치마크 성능을 제공했습니다.
* **효율성**: MSRNet은 완전 합성곱 네트워크이므로 테스트 단계에서 매우 효율적입니다. 400x300 픽셀 이미지에 대해 salient 영역 또는 윤곽선 감지에 0.6초가 소요되며, MCG가 병목이 되어 전체 salient 인스턴스 분할에는 20초가 소요됩니다.

## 🧠 Insights & Discussion

* **인스턴스 수준 이해의 중요성**: 기존의 salient 객체 감지가 단순히 "두드러지는 픽셀"을 찾는 데 그쳤다면, 본 연구는 이를 넘어 "두드러지는 개별 객체 인스턴스"를 식별함으로써 이미지 이해의 수준을 한 단계 높였습니다. 이는 이미지 캡셔닝, 멀티라벨 이미지 인식, 약지도/비지도 학습과 같은 고도화된 응용 분야에 필수적입니다.
* **다중 스케일 및 정제 아키텍처의 효과**: MSRNet의 성공은 다중 스케일 처리와 상향식-하향식 정보 통합을 통한 정제 메커니즘이 salient 객체 감지 및 윤곽선 감지에 매우 효과적임을 입증합니다. 특히, 학습된 주의 모듈은 다양한 스케일의 정보를 지능적으로 융합하여 미세한 디테일까지 포착하는 데 기여합니다.
* **새로운 벤치마크의 가치**: "salient 인스턴스 분할"이라는 새로운 문제를 정의하고, 이에 대한 고품질 주석 데이터셋을 구축한 것은 해당 분야의 후속 연구를 촉진하는 데 중요한 기반을 마련했습니다. 이는 객체 카테고리에 구애받지 않는 일반적인 객체 감지 및 분할 문제 해결에 기여할 수 있습니다.
* **CRF의 역할**: 최종적으로 CRF를 사용하여 객체 제안과 salient 영역 간의 불일치를 해소하고 공간적 일관성을 강화함으로써 분할 품질을 향상시키는 것이 효과적임을 보여줍니다.

## 📌 TL;DR

이 논문은 이미지에서 두드러지는 개별 객체 인스턴스를 식별하는 새로운 과제인 **인스턴스 수준 salient 객체 분할**을 제안합니다. 이를 위해 **MSRNet**이라는 다중 스케일 정제 네트워크를 개발하여 고품질 salient 영역 마스크와 객체 윤곽선을 생성하고, 이를 MCG 및 MAP-기반 최적화와 결합하여 salient 인스턴스 제안을 생성합니다. 마지막으로 CRF를 통해 분할 결과를 정제합니다. 또한, 이 새로운 과제 연구를 위한 **1,000개 이미지의 픽셀 단위 salient 인스턴스 주석 데이터셋**을 구축했습니다. 실험 결과 MSRNet은 모든 공개 salient 영역 감지 벤치마크에서 최첨단 성능을 달성했으며, 새로운 데이터셋에서도 강력한 인스턴스 분할 성능을 입증했습니다.
