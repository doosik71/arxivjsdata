# Fully Convolutional Instance-aware Semantic Segmentation
Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei

## 🧩 Problem to Solve
기존 FCN(Fully Convolutional Networks)은 픽셀별 의미론적 분할(semantic segmentation)에 효과적이지만, 객체 인스턴스(individual object instances)를 구별하는 인스턴스 인식 의미론적 분할(instance-aware semantic segmentation)에는 한계가 있습니다. FCN의 번역 불변성(translation invariance) 특성 때문에 동일한 픽셀이 다른 인스턴스에서 다른 의미를 가질 수 있음에도 불구하고 동일한 응답을 생성합니다.

또한, 기존의 인스턴스 인식 분할 방법론(예: MNC)은 다음과 같은 단점을 가집니다:
1.  ROI(Region of Interest) 풀링 단계에서 특징 왜곡 및 크기 조정으로 인해 공간 세부 정보가 손실됩니다.
2.  마지막 FC(fully-connected) 레이어가 과도하게 파라미터화되어 지역 가중치 공유의 이점을 활용하지 못하고, 특히 마스크 예측을 위해 고차원 출력을 생성할 때 비효율적입니다.
3.  ROI별 네트워크 계산이 ROI 간에 공유되지 않아, 다수의 ROI를 처리할 때 속도가 매우 느립니다.

## ✨ Key Contributions
*   인스턴스 인식 의미론적 분할을 위한 최초의 완전 컨볼루션(fully convolutional) 종단 간(end-to-end) 솔루션인 FCIS(Fully Convolutional Instance-aware Semantic Segmentation)를 제안했습니다.
*   새로운 "위치 감지 내부/외부 스코어 맵(position-sensitive inside/outside score maps)" 개념을 도입하여 객체 분할 및 감지 두 하위 작업을 파라미터 추가 없이 단일 통합 네트워크에서 동시에 수행합니다.
*   기존 ROI 풀링의 단점인 특징 왜곡 및 크기 조정 없이 ROI 내에서 스코어 맵을 재조합(assembling)하여 공간 세부 정보를 보존합니다.
*   정확도와 효율성 모두에서 최첨단 성능을 달성했으며, COCO 2016 분할 경쟁에서 압도적인 차이로 우승했습니다 (2위 대비 12%p 정확도 향상).
*   이전 COCO 2015 우승 모델인 MNC 대비 6배 빠른 추론 속도(이미지당 0.24초)와 약 4배 빠른 학습 속도를 달성했습니다.

## 📎 Related Works
*   **의미론적 이미지 분할(Semantic Image Segmentation):** FCN [29]을 기반으로 한 다양한 접근 방식들 (글로벌 컨텍스트 [28], 멀티스케일 특징 융합 [4], 디컨볼루션 [31] 등).
*   **객체 세그먼트 제안(Object Segment Proposal):** 전통적인 MCG [1], Selective Search [41]와 딥러닝 기반의 DeepMask [32], SharpMask [33] 등이 있으며, 본 논문의 영감이 된 InstanceFCN [5]이 있습니다.
*   **인스턴스 인식 의미론적 분할(Instance-aware Semantic Segmentation):** SDS [15], Hypercolumn [16], CFM [7], MNC [8], MultiPathNet [42] 등 대부분의 최신 방법들은 분할과 감지 작업을 분리하여 순차적으로 수행합니다.
*   **객체 감지를 위한 FCN(FCNs for Object Detection):** InstanceFCN [5]의 "위치 감지 스코어 맵" 아이디어를 객체 감지에 적용한 R-FCN [9].

## 🛠️ Methodology
*   **네트워크 아키텍처:** ResNet-101 [18]을 기본 컨볼루션 네트워크로 사용하며, 마지막 FC 레이어를 제거하고 $1 \times 1$ 컨볼루션 레이어를 추가하여 특징 채널을 1024로 줄입니다.
*   **특징 스트라이드 감소:** `conv5` 레이어의 첫 번째 블록에서 스트라이드를 2에서 1로 줄이고, `conv5`의 모든 컨볼루션 레이어에 dilation 2를 적용하는 "홀 알고리즘(hole algorithm)"($Algorithme \text{`a trous}$) [3, 29]을 사용하여 효과적인 특징 스트라이드를 32에서 16으로 줄여 더 미세한 공간 해상도를 유지합니다.
*   **위치 감지 스코어 맵 생성:** `conv5` 특징 맵 위에 $1 \times 1$ 컨볼루션 레이어를 추가하여 $2k^2 \times (C+1)$ 크기의 스코어 맵을 생성합니다 (여기서 $k=7$은 객체의 $k \times k$ 셀에 해당하며, $C$는 객체 카테고리 수, 1은 배경). 각 스코어 맵은 픽셀이 객체 내 특정 상대적 위치에 속할 확률을 나타냅니다.
*   **마스크 예측 및 분류의 공동 공식화:**
    *   각 픽셀에 대해 "내부(inside)" 스코어와 "외부(outside)" 스코어 두 가지를 예측합니다.
    *   **감지(Detection):** ROI 내 모든 픽셀의 내부/외부 스코어에 대해 픽셀별 `max` 연산을 수행한 후, 모든 카테고리에 걸쳐 평균 풀링 및 소프트맥스(softmax)를 적용하여 ROI의 최종 카테고리 가능성을 계산합니다.
    *   **분할(Segmentation):** 각 픽셀에 대해 내부/외부 스코어에 소프트맥스를 적용하여 픽셀별 전경 마스크 확률을 얻습니다.
    *   두 스코어 세트는 단일 $1 \times 1$ 컨볼루션 레이어에서 파생되며, 분할 및 감지 손실 모두로부터 역전파 그라디언트를 받아 공동으로 학습됩니다.
*   **ROI 생성 및 정제:** RPN(Region Proposal Network) [34]을 사용하여 ROI를 생성하고, 바운딩 박스(bbox) 회귀를 통해 ROI의 위치와 크기를 정제합니다.
*   **학습:**
    *   세 가지 손실 항을 동일한 가중치로 사용합니다: $C+1$ 카테고리에 대한 소프트맥스 감지 손실, 실제 카테고리의 전경 마스크에 대한 소프트맥스 분할 손실, bbox 회귀 손실 [12].
    *   온라인 하드 예제 마이닝(Online Hard Example Mining, OHEM) [38]을 적용하여 각 미니 배치에서 손실이 높은 128개의 ROI를 선택하여 역전파합니다.
*   **추론:** RPN에서 생성된 300개의 ROI를 bbox 회귀로 정제합니다. NMS(Non-Maximum Suppression, IoU 임계값 0.3)로 중복 ROI를 제거한 후, 마스크 투표(mask voting) [8]를 통해 최종 전경 마스크를 생성합니다 (IoU 0.5 이상인 ROI들의 마스크를 가중 평균).

## 📊 Results
*   **PASCAL VOC 2012:**
    *   FCIS는 `mAP_r@0.5`에서 65.7%, `mAP_r@0.7`에서 52.1%를 달성하여 naive MNC, InstFCN + R-FCN, FCIS(translation invariant), FCIS(separate score maps) 등 모든 대안 및 변형 모델보다 우수한 성능을 보였습니다.
    *   이를 통해 위치 감지 스코어 맵과 마스크 예측 및 분류의 공동 공식화가 중요함이 검증되었습니다.
*   **COCO test-dev:**
    *   **MNC 대비:** OHEM 미적용 시 FCIS는 `mAP_r@[0.5:0.95]`에서 28.8%를 달성하여 MNC(24.6%)보다 4.2%p 높았고, 특히 대형 객체에서 정확도 개선이 두드러졌습니다.
    *   **속도:** FCIS는 MNC보다 추론에서 약 6배(이미지당 0.24초 vs 1.37초), 학습에서 약 4배 빨랐습니다. OHEM 적용 시에도 속도 저하 없이 정확도(29.2%)를 높였습니다.
*   **네트워크 깊이:** ResNet-50, ResNet-101, ResNet-152를 사용한 실험에서 ResNet-101 사용 시 ResNet-50보다 성능이 향상되었고, ResNet-152에서는 성능이 포화되는 경향을 보였습니다.
*   **COCO 2016 Segmentation Challenge 우승:** FCIS를 기반으로 멀티스케일 테스트, 수평 뒤집기, 멀티스케일 학습, 6개 네트워크 앙상블을 적용하여 `mAP_r@[0.5:0.95]` 37.6%를 달성했습니다. 이는 2위(G-RMI, 33.8%)와 2015년 우승자(MNC+++, 28.4%)를 크게 능가하는 결과입니다.
*   **COCO Object Detection:** 인스턴스 마스크의 바운딩 박스를 사용하여 객체 감지에서 `mAP_b@[0.5:0.95]` 39.7%를 달성, COCO 객체 감지 리더보드에서 2위를 기록했습니다.

## 🧠 Insights & Discussion
FCIS는 완전 컨볼루션 구조를 통해 기존 FCN의 장점(단순성, 효율성, 지역 가중치 공유)을 인스턴스 인식 분할 작업으로 성공적으로 확장했습니다. "위치 감지 내부/외부 스코어 맵"과 두 하위 작업(분할 및 감지)의 혁신적인 공동 공식화는 파라미터 추가 없이 두 작업 간의 강한 상관관계를 효과적으로 활용하여 정확도와 효율성을 동시에 높였습니다.

FCIS는 기존 ROI 풀링 과정에서 발생하던 공간 정보 손실과 FC 레이어의 과도한 파라미터화 문제를 해결하여, 특히 대형 객체의 분할 정확도를 크게 향상시켰습니다. 또한, 거의 무시할 수 있는 ROI별 계산 비용 덕분에 OHEM과 같은 효과적인 학습 기법을 부담 없이 적용할 수 있었습니다. 본 연구는 인스턴스 인식 분할 분야의 최신 기술을 크게 발전시켰을 뿐만 아니라, 객체 감지 분야에서도 뛰어난 성능을 입증했습니다.

## 📌 TL;DR
**문제:** 기존 FCN은 인스턴스 인식 분할에 부적합하고, 기존 인스턴스 분할 방법은 속도가 느리고 공간 정보 손실이 컸습니다.
**해결책:** 본 논문은 인스턴스 인식 의미론적 분할을 위한 최초의 완전 컨볼루션 방식인 FCIS를 제안합니다. 이 모델은 "위치 감지 내부/외부 스코어 맵"과 마스크 예측/분류의 새로운 공동 공식화를 통해 객체 분할과 감지 작업을 단일 통합 네트워크에서 파라미터 추가 없이 동시에 수행합니다.
**결과:** FCIS는 COCO 2016 분할 경쟁에서 압도적인 차이로 우승했으며, 기존 최신 모델 대비 정확도와 속도 모두에서 큰 폭으로 개선된 최첨단 성능을 달성했습니다 (예: MNC 대비 6배 빠른 추론 속도 및 12%p 이상 정확도 향상).