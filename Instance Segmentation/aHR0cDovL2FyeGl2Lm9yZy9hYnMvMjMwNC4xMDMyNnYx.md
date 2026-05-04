# Ensembling Instance and Semantic for Panoptic Segmentation

Mehmet Yildirim, Yogesh Langhe, Jan Richter, Nanzhu Jiang, Yaroslav Tarasenko, Lyubov Klein, Claus Bahlmann (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Panoptic Segmentation으로, 이는 이미지 내의 모든 픽셀에 대해 클래스 레이블(Category Label)과 인스턴스 ID(Instance Index)를 동시에 부여하는 고도의 장면 이해(Scene Understanding) 작업이다.

Panoptic Segmentation은 크게 두 가지 하위 작업의 결합으로 볼 수 있다. 첫째는 'Thing' 클래스에 대해 객체의 경계를 찾고 개별 객체를 구분하는 Instance Segmentation이며, 둘째는 'Stuff' 클래스(배경, 도로, 하늘 등)를 포함하여 모든 픽셀의 범주를 분류하는 Semantic Segmentation이다. 이 두 작업은 각각의 한계가 존재하며, 이를 통합하여 밀도 있고 일관된 장면 분할 결과를 생성하는 것이 본 연구의 목표이다. 특히 자율 주행과 같은 실시간 환경 인식 시스템에서 이러한 정밀한 장면 이해는 매우 중요하다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Instance Segmentation과 Semantic Segmentation을 각각 독립적으로 수행한 후, 이를 최적으로 결합하는 앙상블 전략을 사용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Expert Models 도입**: 학습 데이터 내의 클래스 불균형(Class Imbalance) 문제를 해결하기 위해, 데이터셋을 빈도수가 높은 클래스(사람, 자동차)와 그 외 클래스로 나누어 각각 최적화된 전문가 모델(Expert Models)을 학습시켰다.
2. **Semantic Ensemble 전략**: 다양한 백본(Backbone)을 가진 여러 개의 DeepLabv3 모델을 학습시키고, 각 모델의 픽셀 단위 신뢰도 맵(Confidence Map)을 평균 내는 앙상블 방식을 통해 Semantic Segmentation의 정확도를 높였다.
3. **고성능 프레임워크 적용**: Mask R-CNN 외에도 Hybrid Task Cascade (HTC) 프레임워크를 도입하여 Instance Segmentation의 성능을 극대화하였으며, 이것이 최종 Panoptic Quality (PQ) 향상에 결정적인 영향을 미침을 보였다.

## 📎 Related Works

본 연구는 기존의 표준적인 Panoptic Segmentation 접근 방식인 [8] (Kirillov et al.)의 프레임워크를 기반으로 한다.

- **Instance Segmentation**: 객체 검출을 위한 Faster R-CNN을 확장하여 마스크 예측 브랜치를 추가한 Mask R-CNN과, R-CNN의 단계를 계층적으로 구성하여 더 정밀한 검출을 가능하게 하는 Hybrid Task Cascade (HTC)를 사용하였다.
- **Semantic Segmentation**: Atrous Convolution과 Spatial Pyramid Pooling (SPP)을 결합하여 다양한 스케일의 정보를 캡처하는 DeepLabv3와 기본 구조인 FCN (Fully Convolutional Networks)을 비교 분석하였다.

기존의 통합 모델(Unified Approach)과 달리, 본 논문은 두 작업을 분리하여 수행함으로써 각 분야의 최신 알고리즘과 개선 사항을 유연하게 적용할 수 있다는 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
전체 시스템은 $\text{Instance Segmentation} \rightarrow \text{Semantic Segmentation} \rightarrow \text{Panoptic Fusion}$의 순서로 진행된다. 최종 결합 단계에서는 `panopticAPI`를 사용하여 두 모델의 예측 결과를 통합한다.

### Instance Segmentation 방법론
1. **Mask R-CNN 및 백본 선택**: ResNet50과 ResNet152를 비교 분석한 결과, ResNet152가 더 높은 성능을 보여 이를 기본 백본으로 사용하였다.
2. **Expert Models 구성**: 데이터 불균형을 해결하기 위해 `train2017` 데이터셋을 세 가지 서브셋(사람 전용, 자동차 전용, 그 외 클래스 전용)으로 분할하고, 각 서브셋에 대해 독립적인 Mask R-CNN 모델을 학습시켰다.
3. **HTC 프레임워크**: 더 높은 정밀도를 위해 $X\text{-}101\text{-}64x4d\text{-}FPN$ 백본과 백본의 마지막 단계(res5)에 Deformable Convolution을 적용한 HTC를 도입하였다.

### Semantic Segmentation 방법론
1. **모델 선택**: FCN보다 DeepLabv3의 성능이 월등히 높음을 확인하여 DeepLabv3를 채택하였다.
2. **Merged-thing 전략**: 학습 시 모든 'Thing' 클래스를 하나의 'merged-thing' 카테고리로 통합하여 처리하였다.
3. **앙상블 절차**: ResNet101 및 ResNet152 백본을 가진 여러 DeepLabv3 모델을 학습시킨 후, 다음과 같이 픽셀 단위 신뢰도 맵의 평균을 구하여 최종 예측을 수행한다.
   $$\text{Final Map} = \frac{1}{N} \sum_{i=1}^{N} \text{Confidence Map}_i$$

### 학습 절차
- **Optimizer**: 모든 실험에서 Momentum이 적용된 Stochastic Gradient Descent (SGD)를 사용하였다.
- **Learning Rate (LR)**: Instance Segmentation은 Warm-up이 포함된 Multi-step 스케줄러를 사용하였으며, Semantic Segmentation은 승수 계수(Multiplicative factor)를 이용한 감쇠(Decaying) 곡선을 적용하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: MS COCO 2017 (`train2017`, `val2017`, `test-dev`)
- **평가 지표**: Instance $\rightarrow$ mAP (bbox, mask), Semantic $\rightarrow$ mIoU, Panoptic $\rightarrow$ PQ (Panoptic Quality)
- **하드웨어**: AWS EC2 (8x GPU V100) 및 Titan Xp

### 주요 결과
1. **Instance Segmentation (Table 1)**:
   - ResNet152 기반 Mask R-CNN이 ResNet50보다 우수하였다.
   - Expert Models 적용 시 bbox mAP가 $41.3 \rightarrow 42.1$로 상승하였다.
   - **HTC 적용 시 가장 높은 성적**인 bbox $50.7$ mAP, mask $43.9$ mAP를 기록하였다.

2. **Semantic Segmentation (Table 2)**:
   - DeepLabv3가 FCN 대비 약 $6.7$ mIoU 높았다.
   - 단일 모델(ResNet101/152)은 약 $43$ mIoU에서 수렴하였으나, 3개 모델을 앙상블했을 때 **$44.5$ mIoU**로 가장 높은 성능을 보였다.

3. **Panoptic Segmentation (Table 3)**:
   - **Baseline**: Mask R-CNN(ResNet152) + DeepLabv3(ResNet101) $\rightarrow$ PQ $42.7$ (val)
   - **Pan1 (Expert Models 추가)**: PQ $43.7$
   - **Pan2 (Semantic Ensemble 추가)**: PQ $43.2$
   - **Pan3 (둘 다 추가)**: PQ $44.2$
   - **Pan4 (HTC + Semantic Ensemble)**: **PQ $47.1$ (test-dev)** 로 최고 성능 달성.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 Panoptic Segmentation을 위해 개별 도메인의 최적화 전략(전문가 모델, 앙상블, 고성능 프레임워크)을 단계적으로 적용함으로써 성능을 향상시켰다. 특히 HTC 프레임워크가 단순한 Instance Segmentation 성능 향상을 넘어 최종 PQ 값에 매우 결정적인 영향을 미친다는 점을 정량적으로 입증하였다. 또한, 정성적 분석(Figure 1)을 통해 HTC 기반 모델(Pan4)이 사람의 다리와 같은 작은 객체 부분까지 더 정밀하게 탐지함을 확인하였다.

### 한계 및 비판적 해석
1. **계산 복잡도**: 통합 모델(Unified approach)에 비해 Instance와 Semantic 모델을 각각 구동하고, 특히 여러 모델을 앙상블하는 방식은 추론 시간과 메모리 비용을 크게 증가시킨다. 실시간성(Real-time)이 요구되는 자율 주행 환경에서는 치명적인 단점이 될 수 있다.
2. **모델 간 의존성**: 두 독립적인 모델의 결과를 단순히 결합하는 방식이므로, 두 모델이 동시에 잘못 예측한 영역에 대한 보정 메커니즘이 부족하다.
3. **데이터 처리**: 'Thing' 클래스를 'merged-thing'으로 통합하여 Semantic Segmentation을 학습시킨 가정이 실제 복잡한 씬에서 얼마나 유효한지에 대한 심층적인 분석은 부족하다.

## 📌 TL;DR

본 논문은 2019 COCO Panoptic Segmentation 챌린지를 위해 **HTC(Instance)와 DeepLabv3 앙상블(Semantic)을 결합한 파이프라인**을 제안하였다. 클래스 불균형 해결을 위한 **Expert Models**와 **모델 앙상블** 전략을 통해 성능을 끌어올렸으며, 최종적으로 **PQ 47.1**이라는 높은 성적을 거두었다. 이 연구는 통합 모델보다 유연하게 최신 개별 알고리즘을 적용할 수 있는 구조적 이점을 보여주며, 향후 고정밀 장면 분할 연구에 있어 개별 모듈의 최적화가 중요함을 시사한다.