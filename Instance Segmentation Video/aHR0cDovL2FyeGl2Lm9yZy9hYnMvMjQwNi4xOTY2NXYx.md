# PM-VIS+: High-Performance Video Instance Segmentation without Video Annotation

Zhangjing Yang, Dun Liu, Xin Wang, Zhe Li, Barathwaj Anandan, Yi Wu (2024)

## 🧩 Problem to Solve

Video Instance Segmentation (VIS)은 비디오 내의 객체를 탐지(Detection), 분할(Segmentation), 그리고 추적(Tracking)하는 것을 목표로 한다. 기존의 고성능 VIS 모델들은 학습을 위해 방대한 양의 비디오 어노테이션(Video Annotation)에 의존한다. 그러나 비디오 데이터, 특히 각 프레임별 객체 마스크(Mask)를 생성하는 작업은 비용이 매우 많이 들고 시간이 오래 걸리기 때문에, 데이터셋의 규모를 확장하는 데 큰 제약이 된다.

본 논문은 수동으로 작성된 비디오 어노테이션 없이 이미지 데이터셋만을 활용하여 고성능 VIS를 달성하는 것을 목표로 한다. 특히 이미지 기반 모델이 비디오의 시간적 일관성(Temporal Consistency)을 활용하지 못해 발생하는 정확도 저하 문제를 해결하고, 비디오 어노테이션 비용을 완전히 제거하면서도 경쟁력 있는 성능을 내는 방법론을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 데이터셋을 통한 초기 학습과 비디오 데이터에 대한 의사 라벨(Pseudo-label) 생성 및 최적화 파이프라인을 결합하는 것이다. 주요 기여 사항은 다음과 같다.

첫째, $\text{COCO}$와 $\text{ImageNet-bbox}$ 데이터셋만을 사용하여 VIS 모델을 학습시키는 방법론을 제시하였다. $\text{ImageNet-bbox}$를 통해 비디오 데이터셋에 존재하는 다양한 카테고리를 보완하였다.

둘째, $\text{PM-VIS}^+$ 알고리즘을 제안하여 데이터의 어노테이션 타입(Bounding Box 또는 Pixel-level Contour)에 따라 감독 신호(Supervision Signal)를 동적으로 조정하도록 설계하였다.

셋째, 학습되지 않은 비디오 데이터에서 추출한 의사 라벨의 노이즈를 제거하기 위해 semi-supervised VOS 모델인 $\text{DeAOT}$를 이용한 최적화 및 $\text{TopK}$, $\text{PScore}$ 기반의 필터링 메커니즘을 도입하였다.

## 📎 Related Works

VIS 방법론은 크게 오프라인(Offline) 방식과 온라인(Online) 방식으로 나뉜다. 오프라인 방식은 추론 시 미래 프레임을 함께 분석하며, 최근에는 $\text{DETR}$ 기반의 쿼리(Query) 방식 모델들이 주를 이룬다. 온라인 방식은 실시간성이 중요하며 $\text{Mask-Track R-CNN}$이나 $\text{IDOL}$과 같은 모델들이 대표적이다. 본 연구는 $\text{IDOL}$을 베이스라인으로 사용한다.

약하게 지도된(Weakly supervised) VIS 연구로는 $\text{MaskFreeVIS}$나 $\text{FlowIRN}$ 등이 있으며, 이들은 주로 Bounding Box 감독이나 분류 라벨 및 광학 흐름(Optical Flow)을 활용한다. 그러나 기존의 약지도 학습 방식들은 데이터 최적화 단계가 부족하여 충분한 성능을 내지 못하는 한계가 있었다. $\text{PM-VIS}^+$는 이미지 데이터셋 활용과 정교한 의사 라벨 최적화 과정을 통해 이러한 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 (Method Flow)
$\text{PM-VIS}^+$의 전체 워크플로우는 비디오 어노테이션 없이 다음의 3단계 과정으로 진행된다.

1.  **의사 라벨 데이터 생성**: 이미지 데이터셋으로 학습된 $\text{PM-VIS}^+(\text{Image})$ 모델을 사용하여 비디오 데이터에 대해 추론을 수행하고 의사 라벨을 생성한다.
2.  **의사 라벨 데이터 최적화**: 생성된 의사 라벨은 $\text{DeAOT}$ 모델을 통해 시간적 일관성을 부여받아 마스크의 질을 높이며, 이후 $\text{TopK}$ 및 $\text{PScore}$ 필터링을 통해 신뢰도가 낮은 데이터를 제거한다.
3.  **최종 모델 학습**: 최적화된 의사 라벨 데이터를 사용하여 $\text{PM-VIS}^+(\text{Video})$ 모델을 최종 학습시킨다.

### 2. 모델 학습 프로세스 및 동적 감독 (Dynamic Supervision)
$\text{PM-VIS}^+$는 학습 데이터의 어노테이션 유형에 따라 손실 함수를 다르게 적용한다.

*   **Pixel-level 데이터 ($\text{COCO}$)**: 마스크 예측 헤드를 학습시키기 위해 $\text{MaskLoss}$를 사용하며, 동시에 Bounding Box 감독을 위한 $\text{BoxInstLoss}$를 함께 적용한다.
*   **Box-level 데이터 ($\text{ImageNet-bbox}$)**: 픽셀 수준의 정답이 없으므로 마스크 예측 헤드를 학습시키지 않고, $\text{BoxInstLoss}$만을 사용하여 객체 탐지 및 위치 정보만을 학습시킨다.
*   **의사 라벨 비디오 데이터**: 픽셀 수준의 마스크가 존재하지만 품질이 일정하지 않으므로, $\text{MaskLoss}$와 $\text{BoxInstLoss}$를 모두 사용하여 상호 보완적인 감독을 수행한다.

### 3. 비디오 의사 라벨 최적화 전략
단순한 추론 결과는 누락된 객체(Missed instances), 오탐(False positives), 부정확한 경계 등의 문제를 가진다. 이를 해결하기 위해 다음 과정을 거친다.

*   **초기화 및 추적**: 비디오 내에서 예측 점수가 가장 높은 프레임을 키프레임(Keyframe)으로 설정하고, $\text{DeAOT}$ 모델을 통해 해당 프레임으로부터 비디오의 양방향(앞뒤)으로 마스크를 전파(Propagation)하여 전체 프레임의 마스크를 완성한다.
*   **필터링 메커니즘**:
    *   $\text{TopK}$ 필터링: 비디오 내 인스턴스의 평균 예측 점수가 높은 상위 $K$개의 마스크만 선택한다.
    *   $\text{PScore}$ 필터링: 평균 예측 점수가 특정 임계값 $\tau$ 이상인 인스턴스만 유지하고 나머지는 폐기한다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: $\text{YTVIS2019}$, $\text{YTVIS2021}$, $\text{OVIS}$ 데이터셋을 사용하여 검증하였다.
*   **백본 네트워크**: $\text{ResNet-50}$ 및 $\text{Swin-L}$을 사용하였다.
*   **지표**: $\text{AP}$ (Average Precision), $\text{AP}_{50}$, $\text{AP}_{75}$, $\text{AR}_1$, $\text{AR}_{10}$ 등을 측정하였다.

### 2. 주요 결과 및 분석
*   **필터링의 효과**: 필터링 없이 의사 라벨을 직접 학습에 사용했을 때보다 $\text{TopK}(K=4)$ 및 $\text{PScore}(\tau=0.2)$를 적용했을 때 $\text{AP}$가 유의미하게 상승하였다. ($\text{YTVIS2019}$ 기준, 필터링 미적용 $41.2\% \rightarrow$ 적용 시 $44.7\%$)
*   **손실 함수의 영향**: 의사 라벨 학습 시 $\text{MaskLoss}$와 $\text{BoxInstLoss}$를 동시에 사용했을 때 가장 높은 성능을 보였다. 특히 $\text{MaskLoss}$ 단독 사용보다 $\text{BoxInstLoss}$를 함께 사용했을 때 성능이 향상되었는데, 이는 의사 라벨의 픽셀 정밀도가 낮아 박스 수준의 감독이 보완 역할을 했음을 시사한다.
*   **백본 네트워크별 성능**: $\text{Swin-L}$ 백본을 사용했을 때 $\text{ResNet-50}$보다 월등한 성능 향상이 나타났다. 특히 $\text{PM-VIS}^+(\text{Video})$ 모델은 $\text{PM-VIS}^+(\text{Image})$ 모델 대비 모든 데이터셋에서 성능 향상을 보였다. ($\text{YTVIS2019}$ $\text{Swin-L}$ 기준: $54.5\% \rightarrow 57.2\%$)

## 🧠 Insights & Discussion

본 논문은 비디오 어노테이션이라는 고비용의 제약을 이미지 데이터셋의 조합과 의사 라벨 최적화라는 전략으로 효율적으로 해결하였다. 특히 $\text{ImageNet-bbox}$를 통해 $\text{COCO}$에서 부족한 카테고리를 보충하고, VOS 모델인 $\text{DeAOT}$의 추적 능력을 이용해 의사 라벨의 시간적 일관성을 확보한 점이 주효했다.

논의할 점은 최종 모델의 성능이 초기 $\text{PM-VIS}^+(\text{Image})$ 모델의 인식 능력에 크게 의존한다는 것이다. 실험 결과에서 초기 모델의 $\text{AP}$가 $1.5\%$ 차이 날 때 최종 모델에서도 $0.7\%$의 성능 차이가 발생함을 확인하였다. 이는 의사 라벨 생성 단계의 품질이 전체 파이프라인의 상한선을 결정한다는 것을 의미한다.

또한, $\text{ImageNet-bbox}$ 데이터의 표준화 부족으로 인해 일부 카테고리에서 예측 정확도가 떨어지는 문제가 언급되었다. 향후 연구에서는 더 정제된 약지도 데이터셋을 확보하거나, 의사 라벨의 노이즈를 더 정교하게 제거하는 기법이 추가된다면 성능을 더욱 끌어올릴 수 있을 것이다.

## 📌 TL;DR

$\text{PM-VIS}^+$는 수동 비디오 어노테이션 없이 $\text{COCO}$와 $\text{ImageNet-bbox}$ 이미지 데이터셋만을 활용하여 고성능 Video Instance Segmentation을 구현한 방법론이다. $\text{PM-VIS}^+(\text{Image})$ 모델로 생성한 의사 라벨을 $\text{DeAOT}$ 모델로 최적화하고 정교한 필터링($\text{TopK}$, $\text{PScore}$)을 거쳐 최종 모델을 학습시킴으로써, 비디오 어노테이션 비용을 획기적으로 줄이면서도 경쟁력 있는 성능을 달성하였다. 이 연구는 데이터 구축 비용이 높은 VIS 분야에서 효율적인 학습 경로를 제시하였다는 점에서 의의가 있다.