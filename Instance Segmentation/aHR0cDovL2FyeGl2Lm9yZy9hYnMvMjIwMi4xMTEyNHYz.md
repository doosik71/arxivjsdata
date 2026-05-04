# Learning with Free Object Segments for Long-Tailed Instance Segmentation

Cheng Zhang, Tai-Yu Pan, Tianle Chen, Jike Zhong, Wenjin Fu, and Wei-Lun Chao (2022)

## 🧩 Problem to Solve

본 논문은 복잡한 장면에서 수많은 클래스를 대상으로 하는 Instance Segmentation 모델을 구축할 때 발생하는 데이터 불균형, 즉 Long-tailed distribution 문제를 해결하고자 한다. 일반적인 데이터셋에서는 빈번하게 등장하는 클래스(Frequent classes)에 비해 희귀한 클래스(Rare objects)의 학습 샘플이 매우 부족하며, 이는 희귀 클래스에 대한 모델 성능의 급격한 저하로 이어진다.

기존의 연구들은 주로 학습 알고리즘, 손실 함수(Loss function)의 수정, 또는 모델 아키텍처 변경을 통해 이 문제를 해결하려 했다. 그러나 이러한 방법들은 주어진 제한된 데이터 내에서 최적화를 시도하는 것이므로 근본적인 데이터 부족 문제를 해결하기 어렵다. 따라서 본 논문의 목표는 노동 집약적인 데이터 수집과 어노테이션 과정 없이, 희귀 클래스의 학습 샘플 수를 효과적으로 늘려 Instance Segmentation 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Object-centric images**(객체 중심 이미지, 예: ImageNet 또는 Google 이미지 검색 결과)를 활용하여 추가 비용 없이 고품질의 객체 세그먼트(Object segments)를 추출하고 이를 학습에 이용하는 것이다. 저자들은 다음 두 가지 직관에 기반한다.

1. Object-centric 이미지는 일반적으로 단순한 배경 속에 하나의 두드러진(Salient) 객체를 포함하고 있다.
2. 동일한 클래스의 객체들은 서로 유사한 외형을 가지거나 배경과의 대비(Contrast)가 유사한 경향이 있다.

이러한 직관을 바탕으로 제안된 **FreeSeg** 프레임워크는 동일 클래스의 이미지들 사이의 공통적인 전경 영역을 찾아내어 "무료" 객체 세그먼트를 생성하고, 이를 기존의 scene-centric 이미지(예: LVIS 데이터셋)에 Copy-paste 방식으로 합성하여 데이터 증강을 수행한다.

## 📎 Related Works

기존의 Long-tailed object detection 및 instance segmentation 연구는 크게 두 갈래로 나뉜다. 첫 번째는 리샘플링(Re-sampling)이나 비용 민감 학습(Cost-sensitive learning)과 같은 학습 단계의 알고리즘을 개선하는 방법이다. 두 번째는 데이터 증강(Data augmentation)을 통해 성능을 높이는 방법으로, Simple Copy-Paste나 MosaicOS 등이 대표적이다.

특히 MosaicOS는 Object-centric 이미지를 활용하여 Long-tailed 문제를 해결하려 했으나, 이는 픽셀 수준의 마스크가 아닌 pseudo bounding box만을 사용한다는 한계가 있다. 본 논문은 여기서 한 단계 더 나아가, 이미지 co-segmentation과 정교한 랭킹 메커니즘을 통해 픽셀 수준의 고품질 인스턴스 라벨을 생성함으로써 Segmentation 성능을 직접적으로 끌어올린다는 점에서 차별점을 가진다.

## 🛠️ Methodology

FreeSeg 프레임워크는 크게 세 가지 단계(세그먼트 생성 및 정제 $\rightarrow$ 세그먼트 랭킹 $\rightarrow$ 데이터 합성)로 구성된다.

### 1. Generating Object Segments (세그먼트 생성 및 정제)
먼저 ImageNet-22K나 Google 이미지 검색을 통해 각 클래스별로 충분한 양의 Object-centric 이미지를 수집한다. 
- **Raw segments generation**: 동일 클래스 이미지들 간의 공통 전경 영역을 추출하기 위해 **SSM(Spatial and Semantic Modulation)**이라는 co-segmentation 알고리즘을 적용하여 그레이스케일의 raw 세그먼트 맵을 얻는다.
- **Segment refinement**: 생성된 맵을 이진 마스크(Binary mask)로 변환하기 위해 Gaussian 필터를 적용한 후, 전경과 배경 사이의 교차 엔트로피를 최소화하는 **Li thresholding**을 통해 최적의 임계값을 결정한다. 이후 Erosion 및 Dilation 연산을 통해 경계를 매끄럽게 하고, 가장 큰 연결 성분(Largest connected component)만을 남겨 노이즈를 제거한다.

### 2. Learning to Rank the Segments (세그먼트 랭킹)
모든 추출된 세그먼트가 정확한 것은 아니므로, 신뢰할 수 있는 고품질 마스크만을 선택하기 위한 랭킹 메커니즘을 도입한다. 이를 위해 ResNet-50 기반의 이미지 분류기를 학습시키고, **LORE(Localization by Region Removal)** 기법을 통해 pseudo bounding box를 생성하여 비교한다.

- **FreeSegscore**: 일반적인 IoU는 박스와 마스크 모두에 노이즈가 있을 때 성능이 낮게 측정되는 경향이 있다. 이를 보완하기 위해 다음 두 지표의 평균을 사용한다.
    - $\text{IoB (Intersection over Bounding box)}$: 마스크가 박스 내부에 얼마나 포함되는지를 측정한다.
    - $\text{IoM (Intersection over Mask)}$: 박스가 마스크를 얼마나 포함하는지를 측정한다.
    - $\text{FreeSegscore} = \frac{\text{IoB} + \text{IoM}}{2}$
- **Drop rate**: 객체가 이미지 내에서 너무 작거나 가려져 있을 경우 세그먼트의 정확도가 낮을 가능성이 크다. 이를 판별하기 위해 LORE 박스를 제거하기 전과 후의 분류기 신뢰도 변화량을 측정한다.
    - $\text{Drop rate} = \frac{s(c) - s'(c)}{s(c)}$ (여기서 $s(c)$는 제거 전, $s'(c)$는 제거 후의 클래스 $c$에 대한 신뢰도이다.)
- 최종적으로 **FreeSegscore와 Drop rate가 모두 0.5보다 큰** 세그먼트만을 고품질 샘플로 선택한다.

### 3. Data Synthesis for Model Training (데이터 합성)
선택된 고품질 세그먼트들을 LVIS와 같은 scene-centric 이미지의 배경 위에 무작위로 붙여넣는 **Simple Copy-Paste** 증강을 수행한다. 이때 세그먼트를 무작위로 리스케일링하고 수평 뒤집기를 적용하여 외형의 다양성(Appearance diversity)을 높인다.

학습은 2단계로 진행된다. 먼저 FreeSeg로 생성된 합성 데이터로 90K 반복 학습을 수행하여 다양한 특징을 학습시키고, 이후 원래의 LVIS 학습 데이터로 다시 90K 반복 학습을 수행하여 예측 결과의 정확도를 보정한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 지표**: LVIS v1 데이터셋을 사용하며, Rare($AP_r$), Common($AP_c$), Frequent($AP_f$) 클래스별 mAP를 측정한다.
- **기준 모델**: Mask R-CNN 및 MosaicOS를 베이스라인으로 사용하였으며, 백본으로 ResNet-50, ResNet-101, ResNeXt-101을 평가하였다.

### 주요 결과
- **정량적 성능**: FreeSeg를 적용했을 때 모든 백본에서 성능 향상이 나타났으며, 특히 희귀 클래스($AP_r$)에서 매우 큰 폭의 향상이 있었다. 예를 들어, ResNet-50 FPN 기반 MosaicOS의 경우 FreeSeg 적용 시 $AP_r$이 $18.17 \rightarrow 20.23$으로 상승하였다.
- **외형 다양성의 중요성**: 단순히 LVIS 내부의 Ground-truth(GT) 세그먼트를 복사-붙여넣기 하는 것보다, 외부에서 수집한 FreeSeg 세그먼트를 사용하는 것이 더 좋은 성능을 보였다. 이는 기존 Copy-paste가 '문맥적 다양성(Context diversity)'에 집중한 반면, FreeSeg는 '외형적 다양성(Appearance diversity)'을 제공하기 때문이다.
- **랭킹의 효과**: 랭킹 과정 없이 모든 세그먼트를 사용하거나 무작위로 샘플링했을 때보다, 제안된 랭킹 메커니즘을 통해 필터링된 고품질 세그먼트를 사용했을 때 성능이 더 높았다. 이는 Instance Segmentation에서 데이터의 양보다 픽셀 라벨의 품질이 더 중요하다는 것을 시사한다.

## 🧠 Insights & Discussion

본 논문은 추가적인 수동 어노테이션 없이 외부의 Object-centric 이미지를 통해 픽셀 수준의 감독 신호(Supervision)를 생성할 수 있음을 입증하였다. 특히 모델 아키텍처에 의존하지 않는(Model-agnostic) 프레임워크이므로, 어떤 Instance Segmentation 모델에도 적용 가능하다는 강점이 있다.

다만, 본 연구는 외부 이미지 검색 결과와 co-segmentation 모델의 성능에 의존한다는 가정을 가지고 있다. 만약 특정 클래스의 Object-centric 이미지 자체가 매우 부족하거나, 배경과 객체의 구분이 모호한 클래스의 경우 세그먼트 추출 품질이 낮아질 수 있다는 잠재적 한계가 있다. 또한, 합성된 이미지가 실제 자연스러운 이미지 분포와는 차이가 있을 수 있으나, 본 논문에서는 이를 2단계 학습 전략을 통해 효과적으로 완화하였다.

비판적으로 해석하자면, 제안된 방법론은 데이터 증강의 관점에서는 매우 효율적이지만, 결국은 '유사한 외형의 데이터를 많이 넣어주는' 방식이다. 이는 모델이 클래스의 본질적인 특징을 학습하게 돕지만, 매우 복잡한 배경과의 상호작용을 학습해야 하는 경우에는 한계가 있을 수 있다.

## 📌 TL;DR

본 논문은 Long-tailed Instance Segmentation 문제를 해결하기 위해, 외부의 객체 중심 이미지에서 자동으로 고품질 마스크를 추출하여 학습 데이터로 활용하는 **FreeSeg** 프레임워크를 제안한다. Co-segmentation과 정교한 랭킹 메커니즘을 통해 생성된 "무료" 세그먼트들을 기존 이미지에 합성함으로써, 특히 희귀 클래스의 외형 다양성을 획기적으로 높여 성능을 크게 향상시켰다. 이 연구는 수동 라벨링 비용 없이 대규모 데이터를 확보하여 딥러닝 모델의 불균형 문제를 해결할 수 있는 새로운 방향성을 제시한다.