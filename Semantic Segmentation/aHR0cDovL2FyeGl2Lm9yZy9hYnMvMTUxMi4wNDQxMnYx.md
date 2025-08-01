# Instance-aware Semantic Segmentation via Multi-task Network Cascades
Jifeng Dai, Kaiming He, Jian Sun

## 🧩 Problem to Solve
의미론적 분할(Semantic Segmentation)은 빠르게 발전하고 있으나, 기존의 완전 합성곱 신경망(FCNs) 기반 방법들은 개별 객체 인스턴스(instance)를 식별하지 못하며, 픽셀 단위로 카테고리 레이블만 예측합니다. 인스턴스 인식 의미론적 분할(Instance-aware Semantic Segmentation)은 객체 인스턴스를 구별하며 분할하는 문제로, 정확도와 속도 면에서 여전히 큰 도전 과제입니다. 특히, 기존 방법들은 계산 비용이 높은 마스크 제안(mask proposal) 모듈(예: MCG)에 의존하여 추론 속도가 매우 느리다는 한계가 있었습니다. 본 논문은 외부 모듈 없이 순수하게 CNN 기반으로 인스턴스 인식 분할을 정확하고 빠르게 수행하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **다중 작업 네트워크 캐스케이드(Multi-task Network Cascades, MNCs) 제안**: 인스턴스 인식 의미론적 분할을 위해 인스턴스 식별, 마스크 추정, 객체 분류의 세 가지 하위 작업으로 구성된 캐스케이드 구조의 CNN 모델을 제시합니다.
*   **차별화 가능한 RoI 워핑 레이어 개발**: 캐스케이드 구조의 인과 관계(causal relation)로 인해 발생하는 비선형적인 종속성을 해결하기 위해 예측된 경계 상자 좌표에 대해 미분 가능한 RoI 워핑 레이어를 개발했습니다. 이를 통해 복잡한 캐스케이드 모델의 종단 간(end-to-end) 학습을 가능하게 합니다.
*   **단일 단계 학습 프레임워크 구현**: 인과적(causal) 캐스케이드 구조를 위한 종단 간 학습 알고리즘을 개발하여, 모든 단계에서 합성곱 특징(convolutional features)을 공유하며 단일 단계로 모델을 훈련할 수 있게 합니다.
*   **최고 수준의 성능 달성**: PASCAL VOC 데이터셋에서 기존 최고 성능 대비 약 3.0% 높은 63.5%의 평균 정밀도(mAP$_{r}$)를 달성했습니다.
*   **획기적인 추론 속도 개선**: VGG-16 모델 사용 시 이미지당 360ms라는 놀라운 속도로, 기존 시스템보다 약 두 자릿수(two orders of magnitude) 빠른 추론 속도를 보여줍니다.
*   **COCO 2015 Segmentation Competition 1위**: MS COCO 2015 분할 경쟁에서 1위를 차지하며, 깊은 Residual Net (ResNet-101)을 사용하여 뛰어난 확장성과 성능을 입증했습니다.

## 📎 Related Works
*   **객체 탐지(Object Detection)**: R-CNN [10], SPPnet [15], Fast R-CNN [9], Faster R-CNN [26] 등 객체 경계 상자와 카테고리 예측에 중점을 둡니다. 특히 Fast/Faster R-CNN은 특징 공유를 통해 속도 향상을 이루었습니다.
*   **인스턴스 인식 의미론적 분할(Instance-aware Semantic Segmentation)**: SDS [13], Hypercolumn [14], CFM [7] 등이 R-CNN 철학을 기반으로 마스크 수준의 영역 제안(region proposals)을 사용합니다. 이들 방법은 MCG [1]와 같은 계산 비용이 높은 외부 마스크 제안 방식에 의존합니다.
*   **카테고리별 의미론적 분할(Category-wise Semantic Segmentation)**: FCNs [23] 및 그 개선 [5, 31]은 픽셀별 카테고리 예측을 수행하지만, 동일 카테고리 내의 개별 인스턴스는 구별하지 못합니다.

## 🛠️ Methodology
MNC 모델은 이미지를 입력으로 받아 인스턴스 인식 의미론적 분할 결과를 출력하는 세 단계의 캐스케이드 구조를 가집니다. 각 단계는 공유된 합성곱 특징(예: VGG-16의 13개 합성곱 레이어)을 활용합니다.

1.  **Stage 1: 상자 수준 인스턴스 회귀 (Box-level Instances Regression)**
    *   **목표**: 클래스에 구애받지 않는(class-agnostic) 경계 상자 형태의 객체 인스턴스를 제안하고 객체성(objectness) 점수를 예측합니다.
    *   **구조**: 공유된 특징 위에 3x3 합성곱 레이어와 두 개의 1x1 합성곱 레이어(상자 위치 회귀 및 객체/비객체 분류)를 사용하며, Faster R-CNN의 RPN (Region Proposal Network) [26] 구조를 따릅니다.
    *   **손실 함수**: RPN 손실 함수인 $L_1 = L_1(B(\Theta))$를 사용합니다. 여기서 $B$는 예측된 상자 목록입니다.

2.  **Stage 2: 마스크 수준 인스턴스 회귀 (Mask-level Instances Regression)**
    *   **목표**: Stage 1에서 제안된 각 상자에 대해 픽셀 수준의 분할 마스크를 예측합니다. 이 마스크 또한 클래스에 구애받지 않습니다.
    *   **구조**: Stage 1에서 예측된 상자에 대해 RoI 풀링(RoI Pooling) [15, 9]을 적용하여 고정 크기(14x14) 특징을 추출합니다. 이 특징에 두 개의 완전 연결(fc) 레이어를 추가하여 픽셀별 마스크(28x28 해상도)를 회귀합니다.
    *   **손실 함수**: $L_2 = L_2(M(\Theta)|B(\Theta))$를 사용합니다. 여기서 $M$은 예측된 마스크 목록이며, $L_2$는 $M$과 $B$에 모두 종속됩니다.

3.  **Stage 3: 인스턴스 분류 (Categorizing Instances)**
    *   **목표**: 각 인스턴스에 대한 카테고리 점수를 출력합니다.
    *   **구조**: Stage 1의 상자에 RoI 풀링을 적용하고, Stage 2의 마스크 예측으로 특징 맵을 "마스킹"($F_{Mask_i}(\Theta) = F_{RoI_i}(\Theta) \cdot M_i(\Theta)$)하여 예측 마스크의 전경에 초점을 맞춥니다. 마스킹된 특징은 두 개의 4096-d fc 레이어를 거치고, 추가적으로 박스 기반의 특징 경로를 결합하여 $N+1$개의 클래스에 대한 Softmax 분류기를 통과합니다.
    *   **손실 함수**: $L_3 = L_3(C(\Theta)|B(\Theta),M(\Theta))$를 사용합니다. 여기서 $C$는 예측된 카테고리 목록이며, $L_3$는 $B$와 $M$에 모두 종속됩니다.

*   **종단 간 학습 (End-to-End Training)**:
    *   전체 손실 함수는 각 단계 손실의 합 $L(\Theta) = L_1(B(\Theta)) + L_2(M(\Theta)|B(\Theta)) + L_3(C(\Theta)|B(\Theta),M(\Theta))$입니다.
    *   **차별화 가능한 RoI 워핑 레이어**: 예측된 상자 $B_i(\Theta)$의 공간 변환(RoI 풀링 결정)에 대한 기울기를 계산하기 위해 미분 가능한 RoI 워핑 레이어를 개발합니다. 이는 bilinear 보간 함수 $\kappa$를 사용하여 특징 맵을 고정 해상도로 크롭하고 워핑하는 과정이며, 경계 상자의 위치 매개변수에 대한 기울기 계산이 가능합니다. 이 레이어는 Spatial Transformer Networks [18]와 유사한 원리를 가집니다.
    *   마스킹 레이어도 원소별 곱셈으로 구현되어 미분 가능합니다. 모든 구성 요소가 미분 가능하므로 확률적 경사 하강법(SGD)으로 종단 간 학습이 가능합니다.

*   **다단계 캐스케이드 확장 (Cascades with More Stages)**:
    *   Stage 3에 클래스별 경계 상자 회귀 레이어를 추가합니다.
    *   추론 시, Stage 3에서 회귀된 상자를 새로운 제안으로 사용하여 Stage 2와 3를 다시 수행합니다. 이는 사실상 5단계 추론 과정입니다 (Stage 4와 5).
    *   이 5단계 구조를 학습 시에도 일관되게 적용하여 (5단계 캐스케이드 학습) 정확도를 더욱 향상시킵니다.

*   **구현 세부 사항**:
    *   Stage 1에서 생성된 약 10^4개의 상자 중 NMS (IoU 0.7)를 통해 상위 300개의 상자를 Stage 2로 전달합니다.
    *   각 단계의 손실 계산을 위한 긍정/부정 샘플 정의 (IoU 임계값 사용).
    *   ImageNet 사전 학습된 VGG-16 또는 ResNet-101 모델로 공유 합성곱 레이어 초기화.
    *   이미지 중심(image-centric) 학습 프레임워크 사용, 미니 배치당 1 이미지, 256개 앵커(Stage 1), 64개 RoI (Stage 2/3).
    *   추론 시, NMS (box-level IoU 0.3)와 "마스크 투표(mask voting)" (유사 마스크들의 가중 평균)를 통해 최종 마스크를 생성하여 정확도를 향상시킵니다.

## 📊 Results
*   **PASCAL VOC 2012 인스턴스 인식 의미론적 분할**:
    *   **소거 연구 (Ablation Experiments)**:
        *   공유 특징 없이 단계별 훈련 (3단계): 60.2% mAP$_{r}$ (VGG-16).
        *   공유 특징을 사용한 단계별 훈련 (3단계): 60.5% mAP$_{r}$.
        *   종단 간 3단계 캐스케이드 훈련: 62.6% mAP$_{r}$.
        *   종단 간 5단계 캐스케이드 훈련: **63.5% mAP$_{r}$**. (최고 성능 달성)
    *   **SOTA 비교**:
        *   MNC: **63.5% mAP$_{r}$@0.5**, **41.5% mAP$_{r}$@0.7**.
        *   이전 최고 성능 (CFM [7]): 60.7% mAP$_{r}$@0.5, 39.6% mAP$_{r}$@0.7.
    *   **추론 시간**: 이미지당 360ms (Nvidia K40 GPU). 기존 방법(MCG 사용)보다 약 두 자릿수(예: 30초 이상) 빠릅니다.

*   **객체 탐지 (PASCAL VOC 2012 test set, mAP$_{b}$)**:
    *   MNC (마스크 레벨 출력 경계 상자): 70.9%
    *   MNC (박스 레벨 출력 경계 상자): 73.5%
    *   MNC (박스 레벨 출력 경계 상자, VOC 07++12 학습): **75.9%**. Fast/Faster R-CNN [9, 26]보다 훨씬 우수한 성능을 보여주며, 마스크 수준 주석의 효과를 입증합니다.

*   **MS COCO 분할**:
    *   VGG-16: 19.5% mAP@[.5:.95], 39.7% mAP@.5.
    *   ResNet-101: 24.6% mAP@[.5:.95], 44.3% mAP@.5. ResNet-101은 VGG-16 대비 26% (mAP@[.5:.95])의 상대적 성능 향상을 가져왔습니다.
    *   ILSVRC & COCO 2015 경쟁의 COCO 분할 트랙에서 **1위**를 차지했습니다 (28.2%/51.5% - 전역 컨텍스트 모델링, 멀티 스케일 테스트, 앙상블 적용).

## 🧠 Insights & Discussion
*   인스턴스 인식 분할 작업을 세 가지 하위 작업으로 분해하는 전략은 매우 효과적입니다.
*   미분 가능한 RoI 워핑 레이어를 통한 종단 간 학습은 특징 공유를 자연스럽게 유도하고 최적화를 개선하여 정확도 향상에 크게 기여합니다.
*   학습 시 구조와 추론 시 구조의 일관성(예: 5단계 캐스케이드)을 유지하는 것이 추가적인 정확도 향상을 가져옵니다.
*   외부 모듈에 의존하지 않고 효율적인 특징 공유를 통해 기존 시스템 대비 압도적인 추론 속도 개선을 달성했습니다.
*   깊은 모델(ResNet-101)의 이점을 쉽게 활용할 수 있음을 입증하며, 방법론의 확장성을 보여주었습니다.
*   향후 연구에서는 CRF [5]와 같은 기법을 활용하여 인스턴스 마스크의 경계를 더욱 정교하게 다듬는 방법을 탐구할 수 있습니다.

## 📌 TL;DR
**문제**: 기존의 의미론적 분할 방법은 객체 인스턴스를 구별하지 못하며, 인스턴스 인식 분할은 느린 외부 모듈에 의존하여 정확도와 속도 모두에서 한계가 있었습니다.
**방법**: 본 논문은 인스턴스 인식 의미론적 분할을 위한 다중 작업 네트워크 캐스케이드(MNCs)를 제안합니다. 이는 상자 제안, 마스크 회귀, 객체 분류의 세 가지 단계로 구성된 캐스케이드 구조입니다. 핵심은 예측된 상자 좌표를 통해 역전파를 가능하게 하는 **차별화 가능한 RoI 워핑 레이어**를 개발하여, 캐스케이드 전체를 종단 간 학습하고 특징을 자연스럽게 공유할 수 있게 한 것입니다. 추론 시에는 5단계 캐스케이드 방식을 사용하며, 학습 시에도 이를 반영하여 성능을 극대화합니다.
**결과**: MNC는 PASCAL VOC에서 **63.5% mAP$_{r}$**로 최고 정확도를 달성했으며, 기존 시스템보다 약 **두 자릿수 빠른 360ms/이미지**의 추론 속도를 보여주었습니다. 또한, MS COCO 2015 분할 경쟁에서 **1위**를 차지하였고, 객체 탐지에서도 기존 SOTA 시스템을 능가하는 경쟁력 있는 결과를 보였습니다.